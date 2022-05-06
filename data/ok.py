
df_list = [1,2,3,4,5,6]
#  类似映射表
value_index_dict = {value: idx for idx, value in enumerate(df_list)}
#  也可以直接 加 index,df.groupby("user_id").index
print(value_index_dict)

