
# coding: utf-8

# In[1]:

from sklearn.datasets import fetch_mldata


# In[2]:

mnist= fetch_mldata('MNIST original', data_home='./mnist')


# In[5]:

# Le dataset principal qui contient toutes les images
print(mnist.data.shape)


# In[6]:

# Le vecteur d'annotations associé au dataset (nombre entre 0 et 9)
print (mnist.target.shape)


# In[8]:

# Echantillonnage
import numpy as np
sample = np.random.randint(70000, size=5000)
data = mnist.data[sample]
target = mnist.target[sample]


# In[9]:

# Séparer le training/testing set
from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(data, target, train_size=0.8)


# In[11]:

# On crée un premier classifieur 3-NN
from sklearn import neighbors

knn = neighbors.KNeighborsClassifier(n_neighbors=3)
knn.fit(xtrain, ytrain)


# In[12]:

# On teste l'erreur de notre classifieur
error = 1 - knn.score(xtest, ytest)
print('Erreur: %f' % error)


# In[14]:

# Optimisation du score sur les données test
import matplotlib.pyplot as plt
errors = []
for k in range(2,15):
    knn = neighbors.KNeighborsClassifier(k)
    errors.append(100*(1 - knn.fit(xtrain, ytrain).score(xtest, ytest)))
plt.plot(range(2,15), errors, 'o-')
plt.show()


# In[15]:

# On récupère le classifieur le plus performant
knn = neighbors.KNeighborsClassifier(3)
knn.fit(xtrain, ytrain)


# In[16]:

# On récupère les prédictions sur les données test
predicted = knn.predict(xtest)


# In[17]:

# On redimensionne les données sous forme d'images
images = xtest.reshape((-1, 28, 28))


# In[18]:

# On selectionne un echantillon de 12 images au hasard
select = np.random.randint(images.shape[0], size=12)


# In[19]:

# On affiche les images avec la prédiction associée
fig,ax = plt.subplots(3,4)

for index, value in enumerate(select):
    plt.subplot(3,4,index+1)
    plt.axis('off')
    plt.imshow(images[value],cmap=plt.cm.gray_r,interpolation="nearest")
    plt.title('Predicted: {}'.format( predicted[value]) )

plt.show()


# In[20]:

# on récupère les données mal prédites 
misclass = (ytest != predicted)
misclass_images = images[misclass,:,:]
misclass_predicted = predicted[misclass]


# In[21]:

# on sélectionne un échantillon de ces images
select = np.random.randint(misclass_images.shape[0], size=12)


# In[22]:

# on affiche les images et les prédictions (erronées) associées à ces images
for index, value in enumerate(select):
    plt.subplot(3,4,index+1)
    plt.axis('off')
    plt.imshow(misclass_images[value],cmap=plt.cm.gray_r,interpolation="nearest")
    plt.title('Predicted: {}'.format(misclass_predicted[value]) )

plt.show()


# In[ ]:



