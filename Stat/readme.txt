# Here are the core data array of this dataset.

# Used for recommender system

## information.npy
The number of users, items, and interactions.

## usrclaim.npy
User portrait, user claims the needed knowledge point of each chapter to strengthen.

## iteminfo.npy
item's features

# Ingredients

## uitime.npy
Time used by a user to finish an item.

## rela_time.npy
The related finish time of a user to finish a problem set(item) (unit: second).

## uiedge.npy
interaction bipartite graph, 1 stands for (U,I) has an edge, vice versa.

## graphvec.npy
Knowledge vectors of random forest, dimension is knowledge point amount(150)*100

# Intermediate products

## match(K).npy
with K means, added the rules' information to the array. (i.e., with knowledge embedding)

## pn(K).npy
Recommend label, 0 means not recommended, vice versa.

## rating(K).npy
user's rate on items. 0.4 *  + 0.6 * personal knowledge match
