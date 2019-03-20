#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/09/17

@author: Maurizio Ferrari Dacrema
"""


from Data_manager.DataReader import DataReader
import tarfile, os


class ThirtyMusicReader(DataReader):
    """
    https://polimi365-my.sharepoint.com/personal/10050107_polimi_it/_layouts/15/guestaccess.aspx?docid=04e1e3c4f884d4a199f43e1d93c46bbfa&authkey=AdfPDEq8eVaiKBHwFO6wZLU
    """

    DATASET_URL = "https://polimi365-my.sharepoint.com/:u:/g/personal/10322330_polimi_it/EW7gg5cotvhPrVC-GyTbSTUB1pKpJ7-8kjsFBB3x8TmK-A?e=ffwHy8"

    DATASET_SUBFOLDER = "ThirtyMusic/"
    AVAILABLE_ICM = ["ICM_tracks"]



    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER



    def _load_from_original_file(self):
        # Load data from original

        print("ThirtyMusicReader: Loading original data")

        compressed_file_folder = self.DATASET_OFFLINE_ROOT_FOLDER + self.DATASET_SUBFOLDER
        decompressed_file_folder = self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER

        credits_path = "entities/albums.idomaar"
        persons_path =  "entities/persons.idomaar"
        playlist_path = "entities/playlist.idomaar"
        tags_path = "entities/tags.idomaar"
        tracks_path =  "entities/tracks.idomaar"
        users_path =  "entities/users.idomaar"

        events_path =  "relations/events.idomaar"
        love_path =  "relations/love.idomaar"
        sessions_path =  "relations/sessions.idomaar"

        try:

            compressed_file = tarfile.open(compressed_file_folder + "ThirtyMusic.tar.gz", "r:gz")
            compressed_file.extract(tracks_path, path=decompressed_file_folder + "decompressed/")
            compressed_file.extract(events_path, path=decompressed_file_folder + "decompressed/")
            compressed_file.close()


        except (FileNotFoundError, tarfile.ReadError, tarfile.ExtractError):

            print("ThirtyMusicReader: Unable to fild data zip file.")
            print("ThirtyMusicReader: Automatic download not available, please ensure the compressed data file is in folder {}.".format(compressed_file_folder))
            print("ThirtyMusicReader: Data can be downloaded here: {}".format(self.DATASET_URL))

            # If directory does not exist, create
            if not os.path.exists(compressed_file_folder):
                os.makedirs(compressed_file_folder)

            raise FileNotFoundError("Automatic download not available.")


        tracks_path = decompressed_file_folder + "decompressed/" + tracks_path
        events_path = decompressed_file_folder + "decompressed/" + events_path




        print("ThirtyMusicReader: loading ICM_tracks")
        self.ICM_tracks, self.tokenToFeatureMapper_ICM_tracks, self.item_original_ID_to_index = self._load_ICM_tracks(tracks_path, if_new_item = "add")

        print("ThirtyMusicReader: loading URM_events")
        self.URM_all, _, self.user_original_ID_to_index = self._load_URM_events(events_path, if_new_user = "add", if_new_item = "ignore")



        print("ThirtyMusicReader: cleaning temporary files")

        import shutil

        shutil.rmtree(decompressed_file_folder + "decompressed/", ignore_errors=True)

        print("ThirtyMusicReader: loading complete")








    def _load_ICM_tracks(self, tracks_path, if_new_item = "add"):

        from Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix_FilterIDs

        ICM_builder = IncrementalSparseMatrix_FilterIDs(preinitialized_col_mapper = None, on_new_col = "add",
                                                        preinitialized_row_mapper = None, on_new_row = if_new_item)


        fileHandle = open(tracks_path, "r")


        numCells = 0

        for line in fileHandle:

            line = line.split("\t")

            #line[0]
            track_id = line[1]

            #line[2]
            line[3] = line[3].replace(':null', ':"null"')
            line[3] = line[3].replace(': null', ':"null"')
            line[3] = eval(line[3])


            track_duration = line[3]["duration"]
            track_playcount = line[3]["playcount"]
            track_MBID = line[3]["MBID"]
            track_name = line[3]["name"]

            line[4] = line[4].replace(':null', ':"null"')
            line[4] = eval(line[4])

            track_artists_list = line[4]["artists"]
            track_albums_list = line[4]["albums"]
            track_tags_list = line[4]["tags"]

            if track_albums_list == "null":
                track_albums_list = []
            if track_tags_list == "null":
                track_tags_list = []


            token_list = [*track_artists_list, *track_albums_list, *track_tags_list]

            token_list = ["{}_{}".format(new_token["type"], new_token["id"]) for new_token in token_list]

            ICM_builder.add_single_row(track_id, token_list, data=1.0)

            numCells += 1

            if numCells % 100000 == 0 and numCells!=0:
                print("Processed {} tracks".format(numCells))


        fileHandle.close()

        return ICM_builder.get_SparseMatrix(), ICM_builder.get_column_token_to_id_mapper(), ICM_builder.get_row_token_to_id_mapper()




    #
    #
    # def _load_sessions(self, sessions_path, if_new_user = "add", if_new_item = "ignore"):
    #
    #
    #
    #     if if_new_user not in ["add", "ignore", "exception"]:
    #         raise ValueError("DataReader: if_new_user parameter not recognized. Accepted values are 'add', 'ignore', 'exception', provided was '{}'".format(if_new_user))
    #
    #     if if_new_item not in ["add", "ignore", "exception"]:
    #         raise ValueError("DataReader: if_new_item parameter not recognized. Accepted values are 'add', 'ignore', 'exception', provided was '{}'".format(if_new_item))
    #
    #     if if_new_user == "ignore":
    #         if_new_user_get_user_index = "exception"
    #     else:
    #         if_new_user_get_user_index = if_new_user
    #
    #     if if_new_item == "ignore":
    #         if_new_item_get_item_index = "exception"
    #     else:
    #         if_new_item_get_item_index = if_new_item
    #
    #
    #
    #
    #
    #     fileHandle = open(sessions_path, "r")
    #
    #     # Use array as for datasets this big lists would require more than 10GB of RAM
    #     dataBlock = 10000000
    #
    #     values = np.zeros(dataBlock, dtype=np.float64)
    #     rows = np.zeros(dataBlock, dtype=np.int32)
    #     cols = np.zeros(dataBlock, dtype=np.int32)
    #
    #     numCells = 0
    #
    #
    #
    #     for line in fileHandle:
    #
    #         line = line.split("\t")
    #
    #         #line[0]
    #         session_id = line[1]
    #         #line[2]
    #
    #         content = line[3].split(" ")
    #         content[0] = eval(content[0])
    #
    #         num_tracks = content[0]["numtracks"]
    #         play_time = content[0]["playtime"]
    #
    #         content[1] = content[1].replace(':null', ':"null"')
    #
    #         content[1] = eval(content[1])
    #
    #         subjects = content[1]["subjects"]
    #         type = subjects[0]["type"]
    #         user_id = subjects[0]["id"]
    #         user_index = self._get_user_index(user_id, if_new = if_new_user_get_user_index)
    #
    #         # Objects contains tracks
    #         objects = content[1]["objects"]
    #
    #         for new_track in objects:
    #
    #             if numCells % 1000000 == 0 and numCells!=0:
    #                 print("Processed {} cells".format(numCells))
    #
    #
    #             track_type = new_track["type"]
    #             track_id = new_track["id"]
    #
    #             track_playstart = new_track["playstart"]
    #             track_playtime = new_track["playtime"]
    #             track_playratio = new_track["playratio"]
    #
    #             track_action = new_track["action"]
    #
    #             try:
    #                 track_index = self._get_item_index(track_id, if_new = if_new_item_get_item_index)
    #             except KeyError:
    #                 # Go to next line
    #                 print("ThirtyMusicReader: Sessions contains track which is not in ICM: {}. Skipping...".format(track_id))
    #                 continue
    #
    #
    #             if numCells == len(rows):
    #                 rows = np.concatenate((rows, np.zeros(dataBlock, dtype=np.int32)))
    #                 cols = np.concatenate((cols, np.zeros(dataBlock, dtype=np.int32)))
    #                 values = np.concatenate((values, np.zeros(dataBlock, dtype=np.float64)))
    #
    #             rows[numCells] = int(user_index)
    #             cols[numCells] = int(track_index)
    #             values[numCells] = True
    #
    #             numCells += 1
    #
    #
    #     fileHandle.close()
    #
    #     return sps.csr_matrix((values[:numCells], (rows[:numCells], cols[:numCells])), dtype=np.float32)
    #






    def _load_URM_events(self, events_path, if_new_user = "add", if_new_item = "ignore"):


        from Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix_FilterIDs

        URM_builder = IncrementalSparseMatrix_FilterIDs(preinitialized_col_mapper = self.item_original_ID_to_index, on_new_col = if_new_item,
                                                        preinitialized_row_mapper = None, on_new_row = if_new_user)



        fileHandle = open(events_path, "r")

        numCells = 0

        for line in fileHandle:

            line = line.split("\t")

            #line[0]
            event_id = line[1]
            #line[2]

            line[3] = eval(line[3])

            play_time = line[3]["playtime"]

            line[4] = eval(line[4])

            subjects = line[4]["subjects"]
            type = subjects[0]["type"]
            user_id = subjects[0]["id"]


            # Objects contains tracks
            objects = line[4]["objects"]

            track_id_list = [str(new_track["id"]) for new_track in objects]

            URM_builder.add_single_row(user_id, track_id_list, data=1.0)

            numCells += 1

            if numCells % 500000 == 0 and numCells!=0:
                print("Processed {} cells".format(numCells))


        fileHandle.close()

        return URM_builder.get_SparseMatrix(), URM_builder.get_column_token_to_id_mapper(), URM_builder.get_row_token_to_id_mapper()









