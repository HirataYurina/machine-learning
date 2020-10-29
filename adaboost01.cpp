#include <iostream>
#include <math.h>
#include <algorithm>

using namespace std;

// 训练数据表
int train_set[10][2] = {
    {0, 1},
    {1, 1},
    {2, 1},
    {3, -1},
    {4, -1},
    {5, -1},
    {6, 1},
    {7, 1},
    {8, 1},
    {9, -1},
};

// 基学习器(一共9个基学习器)
int learner1(int a[2])
{
    if (a[0] < 1.5)
    {
        return 1;
    }
    else
    {
        return -1;
    }
}

int learner2(int a[2])
{
    if (a[0] < 2.5)
    {
        return 1;
    }
    else
    {
        return -1;
    }
}

int learner3(int a[2])
{
    if (a[0] < 3.5)
    {
        return 1;
    }
    else
    {
        return -1;
    }
}

int learner4(int a[2])
{
    if (a[0] < 4.5)
    {
        return 1;
    }
    else
    {
        return -1;
    }
}

int learner5(int a[2])
{
    if (a[0] > 5.5)
    {
        return 1;
    }
    else
    {
        return -1;
    }
}

int learner6(int a[2])
{
    if (a[0] < 6.5)
    {
        return 1;
    }
    else
    {
        return -1;
    }
}

int learner7(int a[2])
{
    if (a[0] < 7.5)
    {
        return 1;
    }
    else
    {
        return -1;
    }
}

int learner8(int a[2])
{
    if (a[0] < 8.5)
    {
        return 1;
    }
    else
    {
        return -1;
    }
}

int learner9(int a[2])
{
    if (a[0] < 9.5)
    {
        return 1;
    }
    else
    {
        return -1;
    }
}

typedef int (*pfunc)(int a[2]);
int num_iter;

int main(int argc, char const *argv[])
{
    int num_train_set = sizeof(train_set) / sizeof(train_set[0]);
    // 数据集数据权重
    double D[num_train_set];
    // 初始化数据集权重
    for (int i = 0; i < num_train_set; i++)
    {
        D[i] = 1.0 / num_train_set;
    }
    pfunc func_array[9] = {learner1, learner2, learner3, learner4, learner5,
                           learner6, learner7, learner8, learner9};
    //迭代次数
    num_iter = atoi(argv[1]);
    // 每个基分类器的错误率
    double errors[9];
    //每一轮迭代后，选取的最佳学习器的权重
    double alpha[num_iter];
    int best_learner[num_iter];

    //开始训练
    cout << "start training with adaboost: iterate steps is " << num_iter << endl;
    for (int step = 0; step < num_iter; step++)
    {
        for (int i = 0; i < 9; i++)
        {
            errors[i] = 0.0;
            for (int j = 0; j < num_train_set; j++)
            {
                errors[i] += D[j] * ((func_array[i](train_set[j]) * train_set[j][1]) == 1 ? 0 : 1);
            }
        }

        // 得到在当前数据集上的最佳学习器
        double min_error = 1.0;
        int min_error_id;
        for (int i = 0; i < 9; i++)
        {
            if (errors[i] < min_error)
            {
                min_error = errors[i];
                min_error_id = i;
            }
        }
        cout << min_error << endl;
        // 计算Gm(x)系数，对应对应式8.2
        // 分类误差率越大，计算出的Gm（x）系数越小
        // Adaboost算法，采用投票机制，希望表现好的学习器在投标中所占权重较大
        alpha[step] = 0.5 * log((1 - min_error) / min_error);

        best_learner[step] = min_error_id;

        double sum = 0;
        for (int i = 0; i < num_train_set; i++)
        {
            // 更新训练数据集的权重分布，对应式子8.3
            double new_d = D[i] * exp(-1 * alpha[step] * train_set[i][1] * func_array[best_learner[step]](train_set[i]));
            // 对应式8.5
            sum += new_d;
            D[i] = new_d;
        }
        for (int i = 0; i < num_train_set; i++)
        {
            // 更新训练样本的权重分布，对应式8.4
            // 有点类似softmax，提高分类错误样本的权值
            D[i] /= sum;
            cout << D[i] << endl;
        }
    }

    int correct_num = 0;
    // 构建基学习器的线性组合
    for (int i = 0; i < num_train_set; i++)
    {
        double result = 0;
        for (int j = 0; j < num_iter; j++)
        {
            double alpha_learner = alpha[j];
            int best_id = best_learner[j];
            result += alpha_learner * func_array[best_id](train_set[i]);
        }
        // 判断对错
        if ((result > 0 ? 1 : -1) == train_set[i][1])
        {
            correct_num += 1;
        }
    }

    double accuracy = correct_num / double(num_train_set);
    cout << "the accuracy of this adaboost algorithm is " << accuracy << endl;

    return 0;
}
