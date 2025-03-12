# src/analyse/events.py

from typing import Any, Dict
import json
import numpy as np
import pandas as pd
import xarray as xr


def export_dict_flood_events_to_csv(
        dict_flood_events: Dict[str, pd.DataFrame],
        th: Any,
        lt: int,
        verbose: bool = False
    ) -> None:
    """
    Export the dictionary with flood events to a csv file
    with a MultiIndex for keys and events

    :param dict_flood_events: dictionary with flood events
    :param th: return period threshold
    :param lt: lead time
    :param verbose: whether to print some test print-s's
    """
    if not dict_flood_events:
        print('No flood events found')
        return
    df_flood_events = pd.concat(dict_flood_events,
                                names = ['identifier', 'events'])
    print(df_flood_events) if verbose else None
                            # sort the index to make the csv file more readable
    df_flood_events = df_flood_events.sort_index()
                            # add n_flood_events to name to
                            # see difference between files
    n_flood_events = df_flood_events.shape[0]
    print(f'exporting {n_flood_events} flood events to csv')
    
    if th in [95, 98, 99]:
        th_str = f'{th}pc'
    else:
        th_str = f'{th}rp'
    df_flood_events.to_csv(
        f'../data/processed/flood_events/flood_events_au_{lt * 24}lt_{th_str}_n{n_flood_events}.csv',
        sep = ';', decimal = '.')
    
    
def create_flood_mask(
        ds: xr.Dataset,
        threshold: int = 5
    ) -> np.ndarray:
    """
    Creates a boolean mask to identify when the streamflow
    exceeds the return period threshold

    :param ds: dataset with streamflow data
    :param threshold: return period threshold
    :return: boolean mask
    """
    if f'RP_{threshold}' not in ds.attrs and f'pc_{threshold}th' not in ds.attrs:
        raise ValueError(f"No return period key ({threshold}) found for {ds.attrs['identifier']}")
    # if ds.attrs[f'RP_{threshold}'] is None and ds.attrs[f'pc_{threshold}th'] is None:
    #     raise ValueError(f"No return period value ({threshold}) found for {ds.attrs['identifier']}")

    # return a boolean mask where True indicates that the streamflow
    # exceeds the return period threshold for the administrative unit
    if f'RP_{threshold}' in ds.attrs:
        return ds['streamflow'] > ds.attrs[f'RP_{threshold}']
    else:
        return ds['streamflow'] > ds.attrs[f'pc_{threshold}th']


def create_event_dict(
        df: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
    """
    Create a dictionary with flood events per admin unit

    :param df: DataFrame with flood events
    :return: dict with df of flood events per admin unit
    """
    d = {}
    for admin_unit, group in df.groupby('identifier'):
        d[admin_unit] = group.reset_index(drop = True)
    return d


def merge_sequential_flood_events(
        df: pd.DataFrame,
        ds: xr.Dataset,
        action_lifetime: int = 10
    ) -> pd.DataFrame:
    """
    Merge flood events that are within one action lifetime of one another

    :param df: DataFrame with flood events
    :param ds: dataset with streamflow data (for calculation of peak streamflow)
    :param action_lifetime: days allowed between events
    :return: DataFrame with merged flood events
    """
                            # convert the DataFrame to a list of dictionaries
    list_events = [df.iloc[0].to_dict()]
    for idx in range(1, len(df)):
                            # for each flood event, get the current and last event,
                            # check if they are within the action lifetime of one another
                            # and merge them if they are
        current_event = df.iloc[idx]
        last_event = list_events[-1]

        days_between = (pd.Timestamp(current_event['flood_start']) - \
                        pd.Timestamp(last_event['flood_end'])).days
        if days_between <= action_lifetime:
                            # within if-s, merge events
            last_event['flood_end'] = pd.Timestamp(current_event['flood_end'])
            last_event['duration'] = (last_event['flood_end'] - last_event['flood_start']).days + 1
                            # recompute peak_streamflow over the new period
            last_event['peak_streamflow'] = ds['streamflow'].sel(
                issue_time = slice(last_event['flood_start'], last_event['flood_end'])
            ).max().item()
        else:
                            # else, no merge needed
            list_events.append(current_event.to_dict())
    
    return pd.DataFrame(list_events)


def get_admin_unit_name(
        p: str
    ) -> str:
    """
    Gets the admin unit name given a pcode

    :param p: PCODE
    :return: admin unit name
    """
    with open('../data/mappings/PCODE_to_Cercle.json', 'r') as f:
        mapping = json.load(f)
    return mapping.get(p)


def add_admin_unit_name_column(
        df: pd.DataFrame
    ) -> pd.DataFrame:
    """
    Adds a column to a DataFrame with the Cercle name based on the 
    pcode in the 'admin_unit' column

    :param df: DataFrame with 'admin_unit' column
    :return: DataFrame with added 'admin_unit_NAME' column
    """
    with open('../data/mappings/PCODE_to_Cercle.json', 'r') as f:
        mapping = json.load(f)
    df['admin_unit_NAME'] = df['admin_unit'].apply(lambda x: mapping.get(x, None))
    return df


def create_flood_events(
        dict_ds_au: Dict[str, xr.Dataset],
        merge = True,
        threshold: int = 5,
        action_lifetime: int = 10
    ) -> Dict[str, pd.DataFrame]:
    """
    Creates flood events based on the return period threshold.
    A flood event is defined as a consecutive period where the
    streamflow exceeds the given threshold for the return period

    :param dict_ds_au: dict of ds per admin unit
    :param threshold: rp-value to trigger flood event (default is 5)
    :param action_lifetime: days allowed between events (default is 10)
    :return: dict with df of flood events per admin unit
    """
    if threshold not in [1.5, 2, 5, 10, 95, 98, 99]:
        raise ValueError("threshold must be 1.5, 2, 5, 10, 95, 98, 99")
    
    flood_events_per_au = {}    # for each admin unit ds:
    for admin_unit, ds in dict_ds_au.items():
                                # create a boolean mask to identify when
                                # the streamflow exceeds the return period threshold
        flood_mask = create_flood_mask(ds, threshold)
        
        flood_events = []
                                # find the start and end of flood events by checking
                                # consecutive True values in the mask
        flood_start, flood_end = None, None
        for idx, is_flood in enumerate(flood_mask):
            if is_flood and flood_start is None:
                                # start of a new flood event
                flood_start = ds['actual_date'].values[idx]
            elif not is_flood and flood_start is not None:
                                # end of the current flood event
                flood_end = ds['actual_date'].values[idx - 1]
                                # inclusive difference (+1) between the two dates;
                                # if the flood event is only one day, the duration is 1;
                                # and, if duration == 1, the event is ignored, as,
                                # hydrologically, the river is expected to handle it
                duration = pd.Timedelta(flood_end - flood_start).days + 1
                if duration == 1:
                    flood_start, flood_end = None, None
                    continue
                                # store the flood event details
                flood_events.append({
                    'flood_start': flood_start,
                    'flood_end': flood_end,
                    'duration': duration,
                    'peak_streamflow': \
                        ds['streamflow'].sel(issue_time = slice(flood_start,
                                                                flood_end)).max().item()
                })
                                # reset flood_start and flood_end
                flood_start, flood_end = None, None
                                # by now, if flood_start is not None,
                                # it means the flood event is still going
        if flood_start is not None:
            flood_end = ds['actual_date'].values[-1]
            flood_events.append({
                'flood_start': flood_start,
                'flood_end': flood_end,
                'duration': pd.Timedelta(flood_end - flood_start).days + 1,
                'peak_streamflow': \
                    ds['streamflow'].sel(issue_time = \
                                         slice(flood_start, flood_end)).max().item()
            })

                                # store the flood events per admin unit in a df,
                                # and merge sequential flood events when needed
                                # (as they are likely the same event)
        df = pd.DataFrame(flood_events)
        if df.empty:            # if no flood events are found, skip the admin unit
            print('No flood events found for admin unit', admin_unit)
            continue

        if merge and len(flood_events) > 0:       
            df = merge_sequential_flood_events(df, ds, action_lifetime)
        flood_events_per_au[admin_unit] = df

    return flood_events_per_au


def combine_dict_events_to_df(d_events: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Combine the dictionary of events to a single dataframe

    :param d_events: dictionary with events
    :return: dataframe with all events
    """
    column_list = ['identifier', 'flood_start', 'flood_end']
    if not d_events:
        return pd.DataFrame(columns = column_list)
    events_list = []
    for admin_unit, df in d_events.items():
        df = df.copy()
        if 'identifier' not in df.columns:
            df['identifier'] = admin_unit
        events_list.append(df)
    df = pd.concat(events_list, ignore_index = True)
    df = df[column_list]
    df['flood_start'] = pd.to_datetime(df['flood_start'])
    df['flood_end'] = pd.to_datetime(df['flood_end'])
    return df[column_list]