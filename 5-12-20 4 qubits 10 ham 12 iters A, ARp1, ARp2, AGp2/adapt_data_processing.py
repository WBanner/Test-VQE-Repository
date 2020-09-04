import numpy as np
import pandas as pd
import math

adapt_data_df = pd.read_csv('adapt_data_df.csv')
adapt_param_df = pd.read_csv('adapt_param_df.csv')
adapt_op_df = pd.read_csv('adapt_op_df.csv')
adapt_E_df = pd.read_csv('adapt_E_df.csv')

adapt_roto_1_data_df = pd.read_csv('adapt_roto_1_data_df.csv')
adapt_roto_1_param_df = pd.read_csv('adapt_roto_1_param_df.csv')
adapt_roto_1_op_df = pd.read_csv('adapt_roto_1_op_df.csv')
adapt_roto_1_E_df = pd.read_csv('adapt_roto_1_E_df.csv')

adapt_roto_2_data_df = pd.read_csv('adapt_roto_2_data_df.csv')
adapt_roto_2_param_df = pd.read_csv('adapt_roto_2_param_df.csv')
adapt_roto_2_op_df = pd.read_csv('adapt_roto_2_op_df.csv')
adapt_roto_2_E_df = pd.read_csv('adapt_roto_2_E_df.csv')

adapt_roto_3_data_df = pd.read_csv('adapt_roto_3_data_df.csv')
adapt_roto_3_param_df = pd.read_csv('adapt_roto_3_param_df.csv')
adapt_roto_3_op_df = pd.read_csv('adapt_roto_3_op_df.csv')
adapt_roto_3_E_df = pd.read_csv('adapt_roto_3_E_df.csv')

exact_E_df = pd.read_csv('exact_energy_df.csv')

exact_E_df.drop(axis = 1,columns = ['Unnamed: 0'], inplace = True)
adapt_data_df.drop(axis = 1,columns = ['Unnamed: 0'], inplace = True)
adapt_param_df.drop(axis = 1,columns = ['Unnamed: 0'], inplace = True)
adapt_op_df.drop(axis = 1,columns = ['Unnamed: 0'], inplace = True)
adapt_E_df.drop(axis = 1,columns = ['Unnamed: 0'], inplace = True)
adapt_roto_1_data_df.drop(axis = 1,columns = ['Unnamed: 0'], inplace = True)
adapt_roto_1_param_df.drop(axis = 1,columns = ['Unnamed: 0'], inplace = True)
adapt_roto_1_op_df.drop(axis = 1,columns = ['Unnamed: 0'], inplace = True)
adapt_roto_1_E_df.drop(axis = 1,columns = ['Unnamed: 0'], inplace = True)
adapt_roto_2_data_df.drop(axis = 1,columns = ['Unnamed: 0'], inplace = True)
adapt_roto_2_param_df.drop(axis = 1,columns = ['Unnamed: 0'], inplace = True)
adapt_roto_2_op_df.drop(axis = 1,columns = ['Unnamed: 0'], inplace = True)
adapt_roto_2_E_df.drop(axis = 1,columns = ['Unnamed: 0'], inplace = True)
adapt_roto_3_data_df.drop(axis = 1,columns = ['Unnamed: 0'], inplace = True)
adapt_roto_3_param_df.drop(axis = 1,columns = ['Unnamed: 0'], inplace = True)
adapt_roto_3_op_df.drop(axis = 1,columns = ['Unnamed: 0'], inplace = True)
adapt_roto_3_E_df.drop(axis = 1,columns = ['Unnamed: 0'], inplace = True)

exact_E_df = exact_E_df.astype(float)*2
exact_E_ser = pd.Series(exact_E_df['ground energy'])

shape = adapt_E_df.shape
numrows = shape[0]

exact_E_list = exact_E_ser.tolist()

adapt_roto_E_metadata = {'adapt mean E': [], 'R1 mean E': [], 'R2 mean E': [], 'R3 mean E': [], 'adapt std E': [], 'R1 std E': [], 'R2 std E': [], 'R3 std E': []}
adapt_roto_op_metadata ={'num ops same adapt R3': [], 'num ops same R1 R2': [], 'num ops same R2 R3':[], 'basic num ops same adapt R3': [], 'basic num ops same R1 R2': [], 'basic num ops same R2 R3': []}
adapt_roto_param_metadata = {'mean param dif adapt R3': [], 'mean param dif R1 R2': [], 'mean param dif R2 R3': [], 'std param dif adapt R3': [], 'std param dif R1 R2': [], 'std param dif R2 R3': []}
adapt_roto_metadata = {'mean eval time': [], 'mean num evals': [], 'mean final energy': [], 'mean final energy std': [], 'same struct count': [], 'mean param dif': []}


adapt_E_abs_dif = adapt_E_df.sub(exact_E_list,axis=1)
adapt_roto_1_E_abs_dif = adapt_roto_1_E_df.subtract(exact_E_list,axis = 1)
adapt_roto_2_E_abs_dif = adapt_roto_2_E_df.subtract(exact_E_list,axis = 1)
adapt_roto_3_E_abs_dif = adapt_roto_3_E_df.subtract(exact_E_list,axis = 1)

print(adapt_E_abs_dif)

#update energy dictionary
adapt_roto_E_metadata['adapt mean E'] = list(adapt_E_abs_dif.mean(axis=1))
adapt_roto_E_metadata['R1 mean E'] = list(adapt_roto_1_E_abs_dif.mean(axis=1))
adapt_roto_E_metadata['R2 mean E'] = list(adapt_roto_2_E_abs_dif.mean(axis=1))
adapt_roto_E_metadata['R3 mean E'] = list(adapt_roto_3_E_abs_dif.mean(axis=1))
adapt_roto_E_metadata['adapt std E'] = list(adapt_E_abs_dif.std(axis=1))
adapt_roto_E_metadata['R1 std E'] = list(adapt_roto_1_E_abs_dif.std(axis=1))
adapt_roto_E_metadata['R2 std E'] = list(adapt_roto_2_E_abs_dif.std(axis=1))
adapt_roto_E_metadata['R3 std E'] = list(adapt_roto_3_E_abs_dif.std(axis=1))

#update op dictionary

#count number of ops that are same starting from end
ttable_a = adapt_op_df.eq(adapt_roto_3_op_df)
ttable_1 = adapt_roto_1_op_df.eq(adapt_roto_2_op_df)
ttable_2 = adapt_roto_2_op_df.eq(adapt_roto_3_op_df)

adapt_R3_param_dif_df = adapt_param_df.sub(adapt_roto_3_param_df)
R1_R2_param_dif_df = adapt_roto_1_param_df.sub(adapt_roto_2_param_df)
R2_R3_param_dif_df = adapt_roto_2_param_df.sub(adapt_roto_3_param_df)

adapt_R3_param_dif_mod_df = adapt_R3_param_dif_df.mod(np.pi)
R1_R2_param_dif_mod_df = R1_R2_param_dif_df.mod(np.pi)
R2_R3_param_dif_mod_df = R2_R3_param_dif_df.mod(np.pi)

adapt_R3_param_dif_df = adapt_R3_param_dif_df.truediv(adapt_R3_param_dif_mod_df)
R1_R2_param_dif_df = R1_R2_param_dif_df.truediv(R1_R2_param_dif_mod_df)
R2_R3_param_dif_df = R2_R3_param_dif_df.truediv(R2_R3_param_dif_mod_df)

for i in range(0,numrows): 
	if len(ttable_a.iloc[i].value_counts()) and ttable_a.iloc[i].value_counts().sort_index(ascending= False).index[0]:
		adapt_roto_op_metadata['basic num ops same adapt R3'].append(ttable_a.iloc[i].value_counts().sort_index(ascending= False).iloc[0])
	else:
		adapt_roto_op_metadata['basic num ops same adapt R3'].append(0)
	if len(ttable_1.iloc[i].value_counts()) and ttable_1.iloc[i].value_counts().sort_index(ascending= False).index[0]:
		adapt_roto_op_metadata['basic num ops same R1 R2'].append(ttable_1.iloc[i].value_counts().sort_index(ascending= False).iloc[0])
	else:
		adapt_roto_op_metadata['basic num ops same R1 R2'].append(0)
	if len(ttable_2.iloc[i].value_counts()) and ttable_2.iloc[i].value_counts().sort_index(ascending= False).index[0]:
		adapt_roto_op_metadata['basic num ops same R2 R3'].append(ttable_2.iloc[i].value_counts().sort_index(ascending= False).iloc[0])
	else:
		adapt_roto_op_metadata['basic num ops same R2 R3'].append(0)
for i in range(0,numrows): 
	if len(ttable_a.iloc[i].value_counts().index) and ttable_a.iloc[i].value_counts().sort_index(ascending= False).index[0]:
		adapt_roto_op_metadata['num ops same adapt R3'].append(ttable_a.iloc[i].value_counts().sort_index(ascending= False).iloc[0])
	else:
		adapt_roto_op_metadata['num ops same adapt R3'].append(0)
	if len(ttable_1.iloc[i].value_counts().index) and ttable_1.iloc[i].value_counts().sort_index(ascending= False).index[0]:
		adapt_roto_op_metadata['num ops same R1 R2'].append(ttable_1.iloc[i].value_counts().sort_index(ascending= False).iloc[0])
	else:
		adapt_roto_op_metadata['num ops same R1 R2'].append(0)
	if len(ttable_2.iloc[i].value_counts().index) and ttable_2.iloc[i].value_counts().sort_index(ascending= False).index[0]:
		adapt_roto_op_metadata['num ops same R2 R3'].append(ttable_2.iloc[i].value_counts().sort_index(ascending= False).iloc[0])
	else:
		adapt_roto_op_metadata['num ops same R2 R3'].append(0)
	if len(list(ttable_a.iloc[i][ttable_a.iloc[i]==False].index)):
		ttable_a = ttable_a.drop(None, axis=1, index=None, columns = list(ttable_a.iloc[i][ttable_a.iloc[i] == False].index))
		adapt_R3_param_dif_df = adapt_R3_param_dif_df.drop(None, axis=1, index=None, columns = list(ttable_a.iloc[i][ttable_a.iloc[i] == False].index))
	if len(list(ttable_1.iloc[i][ttable_1.iloc[i] == False].index)):
		ttable_1 = ttable_1.drop(None, axis=1, index=None, columns = list(ttable_1.iloc[i][ttable_1.iloc[i] == False].index))
		R1_R2_param_dif_df = R1_R2_param_dif_df.drop(None, axis=1, index=None, columns = list(ttable_1.iloc[i][ttable_1.iloc[i] == False].index))
	if len(list(ttable_2.iloc[i][ttable_2.iloc[i] == False].index)):
		ttable_2 = ttable_2.drop(None, axis=1, index=None, columns = list(ttable_2.iloc[i][ttable_2.iloc[i] == False].index))
		R2_R3_param_dif_df = R2_R3_param_dif_df.drop(None, axis=1, index=None, columns = list(ttable_2.iloc[i][ttable_2.iloc[i] == False].index))
	adapt_roto_param_metadata['mean param dif adapt R3'].append(adapt_R3_param_dif_df.iloc[i].mean())
	adapt_roto_param_metadata['mean param dif R1 R2'].append(R1_R2_param_dif_df.iloc[i].mean())
	adapt_roto_param_metadata['mean param dif R2 R3'].append(R2_R3_param_dif_df.iloc[i].mean())
	adapt_roto_param_metadata['std param dif adapt R3'].append(adapt_R3_param_dif_df.iloc[i].std())
	adapt_roto_param_metadata['std param dif R1 R2'].append(R1_R2_param_dif_df.iloc[i].std())
	adapt_roto_param_metadata['std param dif R2 R3'].append(R2_R3_param_dif_df.iloc[i].std())

#update param dictionary

adapt_roto_2_data_dict = {'hamiltonian': [], 'eval time': [], 'num evals': [], 'ansz length': [], 'final energy': []}

adapt_roto_metadata['mean eval time'].append(adapt_data_df['eval time'].mean())
adapt_roto_metadata['mean eval time'].append(adapt_roto_1_data_df['eval time'].mean())
adapt_roto_metadata['mean eval time'].append(adapt_roto_2_data_df['eval time'].mean())
adapt_roto_metadata['mean eval time'].append(adapt_roto_3_data_df['eval time'].mean())

adapt_roto_metadata['mean num evals'].append(adapt_data_df['num evals'].mean())
adapt_roto_metadata['mean num evals'].append(adapt_roto_1_data_df['num evals'].mean())
adapt_roto_metadata['mean num evals'].append(adapt_roto_2_data_df['num evals'].mean())
adapt_roto_metadata['mean num evals'].append(adapt_roto_3_data_df['num evals'].mean())

adapt_roto_metadata['mean final energy'].append(adapt_data_df['final energy'].mean())
adapt_roto_metadata['mean final energy'].append(adapt_roto_1_data_df['final energy'].mean())
adapt_roto_metadata['mean final energy'].append(adapt_roto_2_data_df['final energy'].mean())
adapt_roto_metadata['mean final energy'].append(adapt_roto_3_data_df['final energy'].mean())

adapt_roto_metadata['mean final energy std'].append(adapt_data_df['final energy'].std())
adapt_roto_metadata['mean final energy std'].append(adapt_roto_1_data_df['final energy'].std())
adapt_roto_metadata['mean final energy std'].append(adapt_roto_2_data_df['final energy'].std())
adapt_roto_metadata['mean final energy std'].append(adapt_roto_3_data_df['final energy'].std())

adapt_roto_metadata['same struct count'].append(adapt_roto_op_metadata['num ops same adapt R3'][-1])
adapt_roto_metadata['same struct count'].append(adapt_roto_op_metadata['num ops same R1 R2'][-1])
adapt_roto_metadata['same struct count'].append(adapt_roto_op_metadata['num ops same R2 R3'][-1])
adapt_roto_metadata['same struct count'].append(0)



adapt_roto_param_metadata_df = pd.DataFrame(adapt_roto_param_metadata)

#something up with mean here
adapt_roto_metadata['mean param dif'].append(adapt_roto_param_metadata_df['mean param dif adapt R3'].mean())
adapt_roto_metadata['mean param dif'].append(adapt_roto_param_metadata_df['mean param dif R1 R2'].mean())
adapt_roto_metadata['mean param dif'].append(adapt_roto_param_metadata_df['mean param dif R2 R3'].mean())
adapt_roto_metadata['mean param dif'].append(0)


adapt_roto_metadata_df = pd.DataFrame(adapt_roto_metadata)
adapt_roto_E_metadata_df = pd.DataFrame(adapt_roto_E_metadata)
adapt_roto_op_metadata_df = pd.DataFrame(adapt_roto_op_metadata)

print(adapt_roto_E_metadata_df.shape)

adapt_meta_file = open("adapt_meta.csv","w+")
adapt_E_meta_file = open("adapt_E_meta.csv","w+")
adapt_op_meta_file = open("adapt_op_meta.csv","w+")
adapt_param_meta_file = open("adapt_param_meta.csv","w+")


adapt_roto_metadata_df.to_csv(adapt_meta_file)
adapt_roto_E_metadata_df.to_csv(adapt_E_meta_file)
adapt_roto_op_metadata_df.to_csv(adapt_op_meta_file)
adapt_roto_param_metadata_df.to_csv(adapt_param_meta_file)

adapt_meta_file.close()
adapt_E_meta_file.close()
adapt_op_meta_file.close()
adapt_param_meta_file.close()
