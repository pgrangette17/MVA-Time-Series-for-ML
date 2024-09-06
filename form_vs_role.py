import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from sklearn.metrics import pairwise_distances

from src.record_manager import RecordManager
import pandas as pd
import warnings


def final_process(rm, PART=2):

    warnings.filterwarnings("ignore")

    activity_record, player_periods, roster, ugp_df = rm.load_activity_data(17985)

    print('Player Periods:')
    print(player_periods)

    print('TIME MI-TEMPS \n')
    # we assume there is no prolongations and no penalties session
    time_start_1 = player_periods[player_periods['type']=='START1']['start_dt']
    time_start_1 = pd.to_datetime(time_start_1, format="%Y%m%d%H%M%S").iloc[0]
    idx_end_1 = player_periods[player_periods['type']=='START2'].index[0]-1
    time_end_1 = player_periods.iloc[idx_end_1, 4]
    time_end_1 = pd.to_datetime(time_end_1, format="%Y%m%d%H%M%S")
    time_start_2 = player_periods[player_periods['type']=='START2']['start_dt']
    time_start_2 = pd.to_datetime(time_start_2, format="%Y%m%d%H%M%S").iloc[0]
    time_end_2 = player_periods.iloc[-1, 4]
    time_end_2 = pd.to_datetime(time_end_2, format="%Y%m%d%H%M%S")

    if PART == 1:
        ugp_part_df = pd.read_csv('data/multivariate_players_1.csv')
        chg_pts = pd.read_csv('data/chg_pts_1.csv')
        time_start = time_start_1
        time_end = time_end_1
    else :
        ugp_part_df = pd.read_csv('data/multivariate_players_2.csv')
        chg_pts = pd.read_csv('data/chg_pts_2.csv')
        time_start = time_start_2
        time_end = time_end_2

    checked_chg_pts = []
    print(type(time_start))


    for i, idx_chg_pt in enumerate(chg_pts.time.values) :
        if len(checked_chg_pts) == 0:
            sub_start_b = time_start
            sub_end_b = pd.Timestamp(ugp_part_df.time.iloc[idx_chg_pt])
        else :
            sub_start_b = pd.Timestamp(checked_chg_pts[-1][1])
            sub_end_b = pd.Timestamp(ugp_part_df.time.iloc[idx_chg_pt])
        before_chg_pt = ugp_part_df.iloc[idx_chg_pt-3000:idx_chg_pt-300].mean(axis=0)
        after_chg_pt = ugp_part_df.iloc[idx_chg_pt+300:idx_chg_pt+3000].mean(axis=0)

        # check whether it is a change point or not
        # conditions : (1) there is more than 5 minutes between sub-parts
        #              (2) the relative error between the mean position over the five previous minute (exept last 30sec)
        #                  and the mean position over the following five minutes (except the first 30sec) is larger than an experimental threshold

        # check condition (1)
        if (sub_end_b - sub_start_b).total_seconds() // 60 < 5 :
            print('Detected change point [{}] is insignificant.'.format(ugp_part_df.time.iloc[idx_chg_pt]))
        else :
            # check condition (2)
            err_rel = np.abs(np.divide(after_chg_pt.values - before_chg_pt.values, before_chg_pt.values))
            err_rel[np.isnan(err_rel)] = 0
            err_rel_mean = np.mean(err_rel)
            if err_rel_mean < 0.3 :
                print('Detected change point [{}] is insignificant.'.format(ugp_part_df.time.iloc[idx_chg_pt]))
            else :
                print('Detected change point [{}] is significant.'.format(ugp_part_df.time.iloc[idx_chg_pt]))
                checked_chg_pts.append([idx_chg_pt, ugp_part_df.time.iloc[idx_chg_pt]])
        

    # if yes, check if there is a permutation -> a role permutation
    for i, [idx_chg_pt, chg_pt] in enumerate(checked_chg_pts) :

        before_chg_pt = ugp_part_df.iloc[idx_chg_pt-3600:idx_chg_pt-950].mean(axis=0)
        after_chg_pt = ugp_part_df.iloc[idx_chg_pt+950:idx_chg_pt+3600].mean(axis=0)
        before_chg_pt = np.reshape(before_chg_pt.values, (-1,1))
        after_chg_pt = np.reshape(after_chg_pt.values, (-1,1))
        distance_pairwise = pairwise_distances(X=after_chg_pt, Y=before_chg_pt, metric='euclidean')
        min_dist = np.min(distance_pairwise, axis=0)
        arg_min_dist = np.argmin(distance_pairwise, axis=0)

        perms_list = dict()
        for idx in range(1, ugp_part_df.columns.size-1) :
            if after_chg_pt[idx] !=0 and idx > 0 :
                perms_list[ugp_part_df.columns.values[idx]] = ugp_part_df.columns.values[arg_min_dist[idx-1]+1]

        perms = list()
        for player in perms_list.keys():
            key = player
            viewed_players = []
            if len(perms) !=0 and np.sum([player in perm for perm in perms]) > 0:
                continue
            while key in perms_list.keys() and perms_list[key] not in viewed_players :
                if len(viewed_players)!= 0 and player == perms_list[key]:
                    viewed_players.append(player)
                    perms.append(viewed_players)
                    break
                viewed_players.append(perms_list[key])
                key = perms_list[key]
        
        if len(perms) == 0 :
            print('Detected change point [{}] is not a role permutation.'.format(checked_chg_pts[i][1]))
            print('Detected change point [{}] is a formation changement.'.format(checked_chg_pts[i][1]))
        else :
            print('Detected change point [{}] is a role permutation with players id : {}.'.format(checked_chg_pts[i][1], perms))

            # Now check if it is also a formation change point
            # decompose the permutation into cycles, align the permutated role
            for perm in perms :
                first_val = after_chg_pt[list(ugp_part_df.columns.values).index(perm[0])-1]
                for j in range(len(perm)-1, 0, -1):
                    after_chg_pt[list(ugp_part_df.columns.values).index(perm[j])-1] = after_chg_pt[list(ugp_part_df.columns.values).index(perm[j-1])-1]
                after_chg_pt[list(ugp_part_df.columns.values).index(perm[-1])-1] = first_val
            # compute the mean relative error and then apply a threshold
            err_rel = np.abs(np.divide(after_chg_pt - before_chg_pt, before_chg_pt))
            err_rel[np.isnan(err_rel)] = 0
            err_rel[np.isinf(err_rel)] = err_rel[0]
            err_rel_mean = np.mean(err_rel)
            if err_rel_mean < 0.6 :
                print('Detected change point [{}] is not a formation change point.'.format(checked_chg_pts[i][1]))
            else :
                print('Detected change point [{}] is a formation change point.'.format(checked_chg_pts[i][1]))
            


