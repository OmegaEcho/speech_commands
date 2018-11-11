# speech_commands
## siamese

当前遇到的问题及解决思路：
1、Q：每次训练结果不一致
   A：weight初始值随机，通过 set_random_seed 固定初始值
2、Q：固定迭代次数发生过拟合无法及时停止
   A：使用EarlyStopping，如果问题1能解决，也可以手工调整迭代次数
3、Q：ABS会丢失差异信息
   A：通过内积或差值代替，但收敛速度变慢，尝试增加迭代次数
4、Q：特征信息提取的不好
   A：使用cnn和gru结合的base网络
