# import numpy as np
# import os
# import matplotlib.pyplot as plt
#
#
# def observ():
#     fig1 = plt.figure()
#     plt.imshow(np.random.random((10, 10)))
#     fig2 = plt.figure()
#     plt.imshow(np.random.random((10, 10)))
#     return fig1, fig2
#
#
# save_path = "/home/julin/Documents/MPM_results/"
# print(type(os.listdir(save_path)))
# if os.listdir(save_path) == []:
#     n = 0
# else:
#     print(np.max(list(map(int, os.listdir(save_path)))))
#     n = np.max(list(map(int, os.listdir(save_path)))) + 1
#
# save_path = save_path + str(n) + "/"
# os.mkdir(save_path)
# os.mkdir(save_path+"plots")
# settings_file = open(save_path+"settings.txt", "w")
# settings_file.write(f'bonjour \n  {n}')
# settings_file.close()
#
# # plt.imshow(np.random.random((10, 10)))
# # plt.savefig(save_path + "plots/" + "image.png")
# # plt.show()
#
# fig1, fig2 = observ()
# fig1.show()
# fig2.show()

file = open("/home/julin/Documents/MPM_results/Malik_Erwan/tableau.txt", 'a')
file.write('blablablab'+'\n')
file.close()