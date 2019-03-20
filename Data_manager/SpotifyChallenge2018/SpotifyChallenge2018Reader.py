#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/09/17

@author: Maurizio Ferrari Dacrema
"""


import numpy as np
import zipfile, os, shutil

from Data_manager.DataReader import DataReader
import pandas as pd


class SpotifyChallenge2018Reader(DataReader):

    DATASET_URL = "https://polimi365-my.sharepoint.com/:u:/g/personal/10322330_polimi_it/Ef77B4q8faxEpflGU-kkALIBcb0HUOxo8ER1u_PZNVTPvw?e=7lAc3u"
    DATASET_SUBFOLDER = "SpotifyChallenge2018/"
    AVAILABLE_ICM = []
    DATASET_SPECIFIC_MAPPER = []




    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER



    def _load_from_original_file(self):
        # Load data from original

        print("SpotifyChallenge2018Reader: Loading original data")

        compressed_file_folder = self.DATASET_OFFLINE_ROOT_FOLDER + self.DATASET_SUBFOLDER
        decompressed_file_folder = self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER


        try:

            dataFile = zipfile.ZipFile(compressed_file_folder + "dataset_challenge.zip")
            URM_path = dataFile.extract("interactions.csv", path=decompressed_file_folder + "decompressed/")

        except (FileNotFoundError, zipfile.BadZipFile):

            print("SpotifyChallenge2018Reader: Unable to fild data zip file.")
            print("SpotifyChallenge2018Reader: Automatic download not available, please ensure the compressed data file is in folder {}.".format(compressed_file_folder))
            print("SpotifyChallenge2018Reader: Data can be downloaded here: {}".format(self.DATASET_URL))

            # If directory does not exist, create
            if not os.path.exists(compressed_file_folder):
                os.makedirs(compressed_file_folder)

            raise FileNotFoundError("Automatic download not available.")



        self.URM_all, self.item_original_ID_to_index, self.user_original_ID_to_index = self._load_URM(URM_path, if_new_user = "add", if_new_item = "add")


        print("SpotifyChallenge2018Reader: cleaning temporary files")

        shutil.rmtree(decompressed_file_folder + "decompressed", ignore_errors=True)

        print("SpotifyChallenge2018Reader: loading complete")





    def _load_URM(self, URM_path, if_new_user = "add", if_new_item = "add"):


        from Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix_FilterIDs

        URM_builder = IncrementalSparseMatrix_FilterIDs(preinitialized_col_mapper = None, on_new_col = if_new_item,
                                                        preinitialized_row_mapper = None, on_new_row = if_new_user,
                                                        dtype=np.int32)



        print("SpotifyChallenge2018Reader: Loading csv")

        df_original = pd.read_csv(filepath_or_buffer=URM_path, sep="\t", header=0,
                        usecols=['pid','tid','pos'],
                        dtype={'pid':np.int32,'tid':np.int32,'pos':np.int32})


        print("SpotifyChallenge2018Reader: Removing duplicated positions, keeping latest")

        df_original = df_original.groupby(['tid', 'pid'], as_index=False )['pos'].max()


        print("SpotifyChallenge2018Reader: Building URM sparse")

        playlists = df_original['pid'].values
        tracks = df_original['tid'].values
        position = df_original['pos'].values +1

        URM_builder.add_data_lists(playlists, tracks, position)



        # URM_builder.add_data_lists(playlists, tracks, np.ones_like(playlists).astype(np.int32))
        #
        # URM_implicit_all = URM_builder.get_SparseMatrix()
        # URM_implicit_all = sps.csr_matrix(URM_implicit_all)
        #
        # URM_implicit = URM_implicit_all.copy()
        #
        # n_playlists, n_tracks = URM_implicit.shape
        #
        # URM_implicit.data [URM_implicit.data > 1] = 0
        # URM_implicit.eliminate_zeros()
        #
        # playlists_with_duplicates_mask = np.ediff1d(URM_implicit.indptr)>0
        #
        #
        # # There may be multiple track-playlist interactions, keep the latest
        # for playlist_id in range(n_playlists):
        #
        #     if not playlists_with_duplicates_mask[playlist_id]:
        #
        #         start_pos = URM_implicit_all.indptr[playlist_id]
        #         end_pos = URM_implicit_all.indptr[playlist_id+1]
        #
        #         track_in_playlist = URM_implicit_all.indices[start_pos:end_pos]
        #         position_in_playlist = URM_implicit_all.data[start_pos:end_pos]
        #
        #     else:
        #
        #         playlist_mask = playlists == playlist_id
        #
        #         track_in_playlist = tracks[playlist_mask]
        #         position_in_playlist = position[playlist_mask]
        #
        #         playlist_counter_track = np.bincount(track_in_playlist)
        #
        #         if playlist_counter_track.max()>1:
        #
        #             duplicate_tracks_mask = playlist_counter_track>1
        #             duplicate_tracks_id_list = duplicate_tracks_mask.nonzero()[0]
        #
        #             for duplicate_track in duplicate_tracks_id_list:
        #
        #                 track_in_playlist_mask = track_in_playlist == duplicate_track
        #                 track_position_in_playlist = position_in_playlist[track_in_playlist_mask]
        #
        #                 track_position_in_playlist_argsort = np.argsort(-track_position_in_playlist)
        #
        #                 positions_to_remove = track_position_in_playlist[track_position_in_playlist_argsort[1:]]
        #
        #                 position_in_playlist[position_in_playlist==positions_to_remove] = 0
        #
        #
        #     URM_builder.add_data_lists([playlist_id]*len(track_in_playlist), track_in_playlist, position_in_playlist)
        #
        #
        #     if playlist_id%10 == 0:
        #         print("Processed {} playlists".format(playlist_id))



        return URM_builder.get_SparseMatrix(), URM_builder.get_column_token_to_id_mapper(), URM_builder.get_row_token_to_id_mapper()




