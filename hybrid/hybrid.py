import random
knn = ['simple peach pie',
       'gluten free chocolate cake',
       'chevy s salsa   original recipe',
       'hooters buffalo shrimp',
       'kittencal s greek moussaka']

svd = ['Outback Copycat Alice Springs Chicken',
       'Ensalada Criolla',
       'Amish Triple Butter Biscuits',
       'One Pot Chicken Bacon Spinach Parmesan Pasta',
       'Thai Peanut Coconut Chicken',
       'Banana Coffee Cake',
       "Nif's Cajun Macaroni And Cheese",
       'Coco Oatmeal Honey Cookies',
       'Creole Watermelon Feta Salad',
       'Grilled Cajun Green Beans']

content_based = [137739.0, 35397.0, 42195.0, 261482.0, 112444.0]

'''
Three input:
1. I might want ingredients...
2. Ingredients that I absolutely don't want
3. How likely I want to see other users' recipe with a taste similar to me today

weight based system on knn, svd, content_based
If three input provided by user:
1. I may be want ingredients: Onions
2. Ingredients that I absolutely don't want: garlic
3. similar taste: 0.6

content_based(onion) -> 0.4
knn() -> 0.3
svd() -> 0.3
Remove result with garlic

result: Content_based_result -> knn_result -> svd_result

----------------------------------------------------------------------------------
If three input provided by user:
1. I may be want ingredients: any ingredients works for me
   -> Random generate from all potential ingredients e.g. tomato, cheese
2. Ingredients that I absolutely don't want: No
3. similar taste: 0.8

knn() -> 0.4
svd() -> 0.4
content_based(tomato, cheese) -> 0.2
** No remove step 

result: knn_result -> svd_result -> content_based_result
'''

# three inputs

# suppose we want at the end around 20 recipe
# this is a feedback from front end
similar_taste_weight = 0.5
knn_weight = similar_taste_weight / 2
svd_weight = similar_taste_weight / 2
content_based_weight = 1 - knn_weight - svd_weight

# call knn
knn = ['simple peach pie',
       'gluten free chocolate cake',
       'chevy s salsa   original recipe',
       'hooters buffalo shrimp',
       'kittencal s greek moussaka']
knn_selected = random.sample(knn, round(len(knn) * knn_weight))
print(knn_selected)
# call svd
svd = ['Outback Copycat Alice Springs Chicken',
       'Ensalada Criolla',
       'Amish Triple Butter Biscuits',
       'One Pot Chicken Bacon Spinach Parmesan Pasta',
       'Thai Peanut Coconut Chicken',
       'Banana Coffee Cake',
       "Nif's Cajun Macaroni And Cheese",
       'Coco Oatmeal Honey Cookies',
       'Creole Watermelon Feta Salad',
       'Grilled Cajun Green Beans']
svd_selected = random.sample(knn, round(len(svd) * svd_weight))
print(svd_selected)
# call content-based
# assume it comes back as name of recipes
content_based = [137739.0, 35397.0, 42195.0, 261482.0, 112444.0]
# cleaned_content_based = [item for item in content_based if ]
#

print(random.choices(list(set(['Coco Oatmeal Honey Cookies'])), k=round(
    len(content_based) * content_based_weight)))

unwanted_ingredients = ['onion', 'garlic']
