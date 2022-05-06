# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def get_model():
    """搭建双塔DNN模型"""

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


model = get_model()
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

history = model.fit(
    x=fit_x_train,
    y=y,
    batch_size=32,
    epochs=5,
    verbose=1
)

# ### 3. 模型的预估-predict
# 输入前5个样本并做预测

inputs = df[["UserID_idx", "Gender_idx", "Age_idx", "Occupation_idx", "MovieID_idx", "Genres_idx"]].head(5)
display(df.head(5))

# 对于（用户ID，召回的电影ID列表），计算相似度分数
model.predict([
    inputs["UserID_idx"],
    inputs["Gender_idx"],
    inputs["Age_idx"],
    inputs["Occupation_idx"],
    inputs["MovieID_idx"],
    inputs["Genres_idx"]
])

# 可以提取模型中的user或movie item 的embedding
user_layer_model = keras.models.Model(
    inputs=[model.input[0], model.input[1], model.input[2], model.input[3]],
    outputs=model.get_layer("user_embedding").output
)

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
df_user_embedding.head()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
