import pickle

favorite = pickle.load( open( "save_smile2.p", "rb" ) )
print(list(favorite.keys())[0])
print(len(favorite))
print(favorite[list(favorite.keys())[0]])
