import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import collections


class DssmModel():
    def __init__(self):
        self.x = None
        self.genre_count = None

    # 1. 读取电影数据集（用户信息、电影信息、评分行为信息）
    def read_data(self):
        # MovieLens 1M Dataset：http://files.grouplens.org/datasets/movielens/ml-1m.zip
        # 描述： https://www.jianshu.com/p/a59ff0dc22a3
        # UserID::Gender::Age::Occupation::Zip - code
        df_user = pd.read_csv("data/ml-1m/users.dat",
                              sep="::", header=None, engine="python", encoding='iso-8859-1',
                              names="UserID::Gender::Age::Occupation::Zip-code".split("::"))
        print(df_user.head())
        # MovieID::Title::Genres
        df_movie = pd.read_csv("data/ml-1m/movies.dat",
                               sep="::", header=None, engine="python", encoding='iso-8859-1',
                               names="MovieID::Title::Genres".split("::"))
        print(df_movie.head())
        # UserID::MovieID::Rating::Timestamp
        df_rating = pd.read_csv("data/ml-1m/ratings.dat",
                                sep="::", header=None, engine="python", encoding='iso-8859-1',
                                names="UserID::MovieID::Rating::Timestamp".split("::"))
        print(df_rating.head())
        return df_user, df_movie, df_rating

    # 计算电影中每个题材的次数
    def genre_counts(self, df_movie):
        genre_count = collections.defaultdict(int)
        for genres in df_movie["Genres"].str.split("|"):
            for genre in genres:
                genre_count[genre] += 1
        self.genre_count = genre_count
        print("genre_count:", len(self.genre_count))
        print("genre_count:", self.genre_count)

    # # 每个电影只保留频率最高（代表性）的电影题材标签
    def get_highrate_genre(self, x):
        genre_count = self.genre_count
        sub_values = {}
        for genre in x.split("|"):
            sub_values[genre] = genre_count[genre]  # {"asc": x, ...}
        # print(sub_values)
        highrate_genre = sorted(sub_values.items(), key=lambda x: x[1], reverse=True)[0][0]
        return highrate_genre

    # #### 给特征列做序列编码
    def add_index_column(self, param_df, column_name):
        value_index_dict = {value: idx for idx, value in enumerate(list(param_df[column_name].unique()))}
        param_df[f"{column_name}_idx"] = param_df[column_name].map(value_index_dict)
        return param_df

    def feature_exec(self, df_user, df_movie, df_rating):
        # 每个电影只保留频率最高（代表性）的电影题材标签
        self.genre_counts(df_movie)
        df_movie["Genres"] = df_movie["Genres"].map(self.get_highrate_genre)
        # 给movie特征列做序列编码
        movie_f = ["MovieID", "Genres"]
        for m in movie_f:
            df_movie = self.add_index_column(df_movie, m)
        print(df_movie.head())

        # 给user特征列做序列编码
        user_f = ["UserID", "Gender", "Age", "Occupation"]
        for f in user_f:
            df_user = self.add_index_column(df_user, f)
        print(df_user.head())

        #
        min_rating = df_rating["Rating"].min()
        max_rating = df_rating["Rating"].max()
        df_rating["Rating"] = df_rating["Rating"].map(
            lambda x: (x - min_rating) / (max_rating - min_rating))  # 评分作为两者的相似度
        # df_rating["is_rating_high"] = (df["Rating"]>=4).astype(int)  # 可生成是否高评分作为分类模型的类别标签

        # 合并成一个df
        df = pd.merge(pd.merge(df_rating, df_user), df_movie)
        df.drop(columns=["Timestamp", "Zip-code", "Title"], inplace=True)
        print("df", df.head())
        return df

    # num_users, num_movies, num_genders, num_ages, num_occupations, num_genres

    # # #### 评分的归一化
    # def rating_deal(self, df):
    #     min_rating = df["Rating"].min()
    #     max_rating = df["Rating"].max()
    #
    #     df["Rating"] = df["Rating"].map(lambda x: (x - min_rating) / (max_rating - min_rating))  # 评分作为两者的相似度
    #     # df["is_rating_high"] = (df["Rating"]>=4).astype(int)  # 可生成是否高评分作为分类模型的类别标签
    #     df.sample(frac=1).head(3)
    #
    #     return df

    # 构建训练集特征及标签
    def train_test_set(self, df_user, df_movie, df_rating):
        df = self.feature_exec(df_user, df_movie, df_rating)
        df_sample = df.sample(frac=0.1)  # 训练集抽样
        X = df_sample[["UserID_idx", "Gender_idx", "Age_idx", "Occupation_idx", "MovieID_idx", "Genres_idx"]]
        y = df_sample["Rating"]
        return df, X, y

    def get_model(self, df):
        """搭建双塔DNN模型"""
        num_users = df["UserID_idx"].max() + 1
        num_movies = df["MovieID_idx"].max() + 1
        num_genders = df["Gender_idx"].max() + 1
        num_ages = df["Age_idx"].max() + 1
        num_occupations = df["Occupation_idx"].max() + 1
        num_genres = df["Genres_idx"].max() + 1

        # 输入
        user_id = keras.layers.Input(shape=(1,), name="user_id")
        gender = keras.layers.Input(shape=(1,), name="gender")
        age = keras.layers.Input(shape=(1,), name="age")
        occupation = keras.layers.Input(shape=(1,), name="occupation")
        movie_id = keras.layers.Input(shape=(1,), name="movie_id")
        genre = keras.layers.Input(shape=(1,), name="genre")

        # user 塔
        user_vector = tf.keras.layers.concatenate([
            layers.Embedding(num_users, 100)(user_id),
            layers.Embedding(num_genders, 2)(gender),
            layers.Embedding(num_ages, 2)(age),
            layers.Embedding(num_occupations, 2)(occupation)
        ])
        user_vector = layers.Dense(32, activation='relu')(user_vector)
        user_vector = layers.Dense(8, activation='relu',
                                   name="user_embedding", kernel_regularizer='l2')(user_vector)

        # item 塔
        movie_vector = tf.keras.layers.concatenate([
            layers.Embedding(num_movies, 100)(movie_id),
            layers.Embedding(num_genres, 2)(genre)
        ])
        movie_vector = layers.Dense(32, activation='relu')(movie_vector)
        movie_vector = layers.Dense(8, activation='relu',
                                    name="movie_embedding", kernel_regularizer='l2')(movie_vector)

        # 每个用户的embedding和item的embedding作点积
        dot_user_movie = tf.reduce_sum(user_vector * movie_vector, axis=1)
        dot_user_movie = tf.expand_dims(dot_user_movie, 1)

        output = layers.Dense(1, activation='sigmoid')(dot_user_movie)

        return keras.models.Model(inputs=[user_id, gender, age, occupation, movie_id, genre], outputs=[output])

    def training(self, df, X, y):
        model = self.get_model(df)

        model.compile(loss=tf.keras.losses.MeanSquaredError(),
                      optimizer=keras.optimizers.RMSprop())
        fit_x_train = [
            X["UserID_idx"],
            X["Gender_idx"],
            X["Age_idx"],
            X["Occupation_idx"],
            X["MovieID_idx"],
            X["Genres_idx"]
        ]

        model.fit(
            x=fit_x_train,
            y=y,
            batch_size=32,
            epochs=10,
            verbose=1
        )

        return model

    def predicting(self, ):
        # ### 3. 模型的预估-predict
        # 输入前5个样本并做预测
        df_user, df_movie, df_rating = self.read_data()
        df, X, y = self.train_test_set(df_user, df_movie, df_rating)
        model = self.training(df, X, y)

        # predicting
        inputs = df[["UserID_idx", "Gender_idx", "Age_idx", \
                     "Occupation_idx", "MovieID_idx", "Genres_idx"]]
        print("inputs:", inputs.head(5))
        # 对于（用户ID，召回的电影ID列表），计算相似度分数
        results = model.predict([
            inputs["UserID_idx"],
            inputs["Gender_idx"],
            inputs["Age_idx"],
            inputs["Occupation_idx"],
            inputs["MovieID_idx"],
            inputs["Genres_idx"]
        ])
        print("results: ", results)

        # 可以提取模型中的user或movie item 的embedding
        user_layer_model = keras.models.Model(
            inputs=[model.input[0], model.input[1], model.input[2], model.input[3]],
            outputs=model.get_layer("user_embedding").output
        )

        # user_id --> embedding  [[user_id, ...], [ , ], [, ], [, ]]
        user_embeddings = []
        for index, row in df_user.iterrows():
            user_id = row["UserID"]
            user_input = [
                np.reshape(row["UserID_idx"], [1, 1]),
                np.reshape(row["Gender_idx"], [1, 1]),
                np.reshape(row["Age_idx"], [1, 1]),
                np.reshape(row["Occupation_idx"], [1, 1])
            ]
            user_embedding = user_layer_model(user_input)
            embedding_str = ",".join([str(x) for x in user_embedding.numpy().flatten()])
            user_embeddings.append([user_id, embedding_str])

        df_user_embedding = pd.DataFrame(user_embeddings, columns=["user_id", "user_embedding"])
        print(df_user_embedding.head(5))


if __name__ == '__main__':
    ss = DssmModel()
    ss.predicting()
