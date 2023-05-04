from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
medicalpca = medical[['Age', 'Children', 'Income', 'VitD_levels', 'Doc_vistits', 'Full_meals_eaten','VitD_supp',
                      'Initial_days', 'TotalCharge', 'Additional_charges', 'Item1','Item2','Item3','Item4','Item5',
                      'Item6','Item7','Item8']] #Create a separate Dataframe that contains only continuous data for PCA analysis

medicalpca_normalized = (medicalpca-medicalpca.mean())/medicalpca.std.() #Normalize the data by taking the value, minus the mean, divided by the standard deviation.
pca = PCA(n_components=medicalpca.shape[1]) #Initialize PCA with a number of components equal to the components in the medicalpca dataframe (in this case 18)
pca.fit(medicalpca_normalized) # fit the normalized data to the pca
medicalpca_pca=pd.DataFrame(pca.transform(medicalpca_normalized), 
                            columns=['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10','PC11','PC12','PC13','PC14',
                                     'PC15','PC16','PC17','PC18'])

plt.plot(pca.explained_variance_ratio_) #Create a Scree plot showing the explained variance over the number of components
plt.xlabel('number of components')
plt.ylabel('explained variance')
plt.show()
#Convert the pca explained variance ratios into eigenvalues to determine how many components need to be used for the model.
cov_matrix = np.dot(medicalpca_normalized.T, medicalpca_normalized)/medicalpca.shape[0]
eigenvalues=[np.dot(eigenvector.T,np.dot(cov_matrix,eigenvector)) for eigenvector in pca.components_]
plt.plot(eigenvalues)
plt.xlabel('number of components')
plt.ylabel('eigenvalue')
plt.show()
