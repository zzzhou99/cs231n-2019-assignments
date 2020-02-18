# 重要参考网站

[Syllabus | CS 231N](http://vision.stanford.edu/teaching/cs231n/syllabus.html)

[CS231n官方笔记授权翻译总集篇](https://zhuanlan.zhihu.com/p/21930884)




# git使用教训备忘

[码云提交代码教程](https://gitee.com/help/articles/4122)

git init

git remote add origin https://gitee.com/zhouzizhen/cs231n-2019-assignments.git

git remote add origin https://github.com/monodrama99/cs231n-2019-assignments.git

git pull origin master

git add .

git commit -m "提交"

git push origin master

在新建仓库时，如果在码云平台仓库上已经存在 readme 或其他文件，在提交时可能会存在冲突，这时用户需要选择的是保留线上的文件或者舍弃线上的文件

如果您舍弃线上的文件，则在推送时选择强制推送，强制推送需要执行下面的命令(默认不推荐该行为)：

git push origin master -f

如果您选择保留线上的 readme 文件,则需要先执行：

git pull origin master

# 一些报错解决

remote: Incorrect username or password ( access token )

![凭据管理器要设置好](https://images.gitee.com/uploads/images/2020/0218/123039_c1cbe4f8_2114684.jpeg "gitee凭据.jpg")

remote: error: GE007: Your push would publish a private email address.

多邮箱管理→不公开我的邮箱地址→不打勾
