{
 cells: [
  {
   cell_type: code,
   execution_count: null,
   metadata: {},
   outputs: [],
   source: []
  },
  {
   cell_type: code,
   execution_count: 1,
   metadata: {
    id: "dMKbDnpmABdF"
   },
   outputs: [],
   source: [
    import numpy as np\n,
    import pandas as pd\n,
    from math import pi,exp,sqrt
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# load and prepare data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "id": "hxskoVXfVcv0"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Real class : {'setosa', 'virginica', 'versicolor'}\n",
      "coded cllass : {0, 1, 2}\n",
      "dataset: 150\n"
     ]
    }
   ],
   "source": [
    "# load data\n",
    "df = pd.read_csv('iris.csv')\n",
    "print(\"Real class :\",set(df['species']))\n",
    "df['species'] =df['species'].astype('category').cat.codes\n",
    "print(\"coded cllass :\",set(df['species']))\n",
    "dataset=df.values\n",
    "print(\"dataset:\",len(dataset))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Shuffle And Split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "from random import shuffle\n",
    "def shuffled(dataset):\n",
    "    shuffled_indices = list(range(len(dataset)))\n",
    "    shuffle(shuffled_indices)\n",
    "    dataset[:]=dataset[shuffled_indices]\n",
    "    return dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "dataset=shuffled(dataset)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "split test and train"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "test: 45\n",
      "train: 105\n"
     ]
    }
   ],
   "source": [
    "test_len=round(0.3*len(dataset))\n",
    "test =dataset[:test_len]\n",
    "train=dataset[test_len:]\n",
    "print(\"test:\",len(test))\n",
    "print(\"train:\",len(train))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# NB \n",
    "$P(C_i|X)P(X)=P(X|C_i)P(C_i)$\n",
    "\n",
    "$P(X|C_i)=P(x_1|C_i)P(x_2|C_i)P(x_3|C_i)P(x_4|C_i)$\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Gaussian PDF"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "from math import sqrt\n",
    "# Calculate the mean of a list of numbers\n",
    "def mean(numbers):\n",
    "    return sum(numbers)/float(len(numbers))\n",
    "# Calculate the standard deviation of a list of numbers\n",
    "def stdev(numbers):\n",
    "    avg = mean(numbers)\n",
    "    variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers)-1)\n",
    "    return sqrt(variance)\n",
    "# Calculate meen_Sigma_len for each class\n",
    "def meen_Sigma_len(dataset):\n",
    "    X=np.array(dataset)\n",
    "    X=X[:,:-1]\n",
    "    m=np.mean(X, axis = 0)\n",
    "    N=len(X[:,0])\n",
    "    sigma=np.dot(np.transpose(X-m),X-m)/(N-1)\n",
    "    return m,sigma,N\n",
    "def mean_std_len(dataset):\n",
    "    summaries = [(mean(column), stdev(column), len(column)) for column in zip(*dataset)]\n",
    "    del(summaries[-1])\n",
    "    return summaries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Calculate the Gaussian probability distribution function for x\n",
    "#P(x|mean,sigma)\n",
    "def probability(row,m,Sig): \n",
    "    row=row[:-1]\n",
    "    S_inv=np.linalg.inv(Sig)\n",
    "    det_S= np.linalg.det(Sig)\n",
    "    exponent = exp(-0.5*np.dot(np.transpose(row-m),np.dot(S_inv,row-m)))\n",
    "    return  ((det_S)**0.5) * exponent\n",
    "#P(x|mean,stdev)\n",
    "def calculate_probability(x, mean, stdev):\n",
    "    exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))\n",
    "    return (1 / (sqrt(2 * pi) * stdev)) * exponent"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# separate and summaries train dataset by class"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "#separate train datasetby class\n",
    "def separate_by_class(dataset):\n",
    "    separated = dict()\n",
    "    for i in range(len(dataset)):\n",
    "        vector = dataset[i]\n",
    "        class_value = vector[-1]\n",
    "        if (class_value not in separated):\n",
    "            separated[class_value] = list()\n",
    "        separated[class_value].append(vector)\n",
    "    return separated\n",
    "# Split dataset by class then calculate statistics for each row\n",
    "def summarize_by_class(dataset):\n",
    "    separated = separate_by_class(dataset)\n",
    "    summaries = dict()\n",
    "    for class_value, rows in separated.items():\n",
    "        summaries[class_value] = mean_std_len(rows)\n",
    "    return summaries\n",
    "# Split dataset by class then calculate statistics for each row\n",
    "def summarize_class(dataset):\n",
    "    separated = separate_by_class(dataset)\n",
    "    summaries = dict()\n",
    "    for class_value, rows in separated.items():\n",
    "        summaries[class_value] =meen_Sigma_len(rows)\n",
    "    return summaries"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# NB"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Calculate the probabilities of predicting each class for a given row\n",
    "def calculate_class_probabilities(summaries, row):  # summaries from train and rowfrom test\n",
    "    total_rows = sum([summaries[label][0][2] for label in summaries])\n",
    "    probabilities = dict()\n",
    "    for class_value, class_summaries in summaries.items():\n",
    "        probabilities[class_value] = summaries[class_value][0][2]/float(total_rows)  #P(Ci)=Ni/N\n",
    "        for i in range(len(class_summaries)):\n",
    "            mean, stdev, _ = class_summaries[i]\n",
    "            probabilities[class_value] *= calculate_probability(row[i], mean, stdev) #P(x|Ci)=P(Ci)P(x1|Ci)P(x2|Ci)...P(xd|Ci)\n",
    "    return probabilities #[P(x|C0),P(x|C1),P(x|C2),...,P(x|Ck)]\n",
    "# Predict the class for a given row\n",
    "def predict(summaries, row):\n",
    "    probabilities = calculate_class_probabilities(summaries, row)\n",
    "    best_label, best_prob = None, -1\n",
    "    for class_value, probability in probabilities.items():\n",
    "        if best_label is None or probability > best_prob: #find Max(P(x|Ci))\n",
    "            best_prob = probability\n",
    "            best_label = class_value\n",
    "    return best_label \n",
    " \n",
    "# Naive Bayes Algorithm\n",
    "def naive_bayes(train, test):\n",
    "    summarize = summarize_by_class(train)\n",
    "    predictions = list()\n",
    "    for row in test:\n",
    "        output = predict(summarize, row)\n",
    "        predictions.append(output)\n",
    "    return np.array(predictions).astype(int)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "def class_probabilities(summaries, row):  # summaries from train and rowfrom test\n",
    "    total_rows = sum([summaries[label][0][2] for label in summaries])\n",
    "    probabilities = dict()\n",
    "    for class_value, class_summaries in summaries.items():\n",
    "        m, sigma, Ni = class_summaries\n",
    "        probabilities[class_value] = Ni/float(total_rows)  #P(Ci)=Ni/N\n",
    "        probabilities[class_value] *= probability(row, m, sigma) #P(Ci|X)=P(Ci)P(X|Ci)\n",
    "    return probabilities #[P(x|C0),P(x|C1),P(x|C2),...,P(x|Ck)]\n",
    "\n",
    "def predicting(summaries, row):\n",
    "    probabilities = class_probabilities(summaries, row)\n",
    "    best_label, best_prob = None, -1\n",
    "    for class_value, probability in probabilities.items():\n",
    "        if best_label is None or probability > best_prob: #find Max(P(x|Ci))\n",
    "            best_prob = probability\n",
    "            best_label = class_value\n",
    "    return best_label \n",
    "# Naive Bayes Algorithm\n",
    "def GNB(train, test):\n",
    "    summarize = summarize_class(train)\n",
    "    predictions = list()\n",
    "    for row in test:\n",
    "        output = predicting(summarize, row)\n",
    "        predictions.append(output)\n",
    "    return np.array(predictions).astype(int)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "y_T: [2 2 0 1 2 0 2 1 1 1 0 0 0 2 2 2 2 0 2 1 0 1 2 0 2 2 0 1 1 2 2 0 2 2 2 1 0\n",
      " 0 0 0 1 1 0 0 2]\n",
      "y_p1: [2 2 0 1 2 0 2 1 1 1 0 0 0 2 2 2 2 0 2 1 0 1 2 0 2 2 0 1 1 2 2 0 2 2 2 1 0\n",
      " 0 0 0 1 1 0 0 2]\n",
      "y_p2: [2 2 0 1 2 0 2 1 1 1 0 0 0 2 2 2 2 0 2 1 0 2 2 0 2 2 0 1 1 2 2 0 2 2 2 1 0\n",
      " 0 0 0 1 1 0 0 2]\n"
     ]
    }
   ],
   "source": [
    "# Test Naive Bayes Algorithm\n",
    "y_p1= naive_bayes(train, test)\n",
    "y_p2= GNB(train, test)\n",
    "y_T=np.array(test[:,-1]).astype(int)\n",
    "print(\"y_T:\",y_T)\n",
    "print(\"y_p1:\",y_p1)\n",
    "print(\"y_p2:\",y_p2)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# accuracy and confusion_matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "id": "0noP3Lw6d9Qr"
   },
   "outputs": [],
   "source": [
    "def conf_matrix(y,y_hat):\n",
    "    M=np.zeros([3,3])\n",
    "    L = len(y)\n",
    "    for i in range(L):\n",
    "        M[y[i],y_hat[i]]+=1 \n",
    "    return M\n",
    "def accuracy(y,y_hat):\n",
    "    count=0\n",
    "    L=len(y)\n",
    "    for i in range(L):\n",
    "        if y[i]==y_hat[i]:\n",
    "            count +=1\n",
    "    return (count/L)\n",
    "def conf_report(M,Tlabels,Plabels):\n",
    "    T=pd.DataFrame(M,index=Tlabels,columns=Plabels)\n",
    "    return T"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "accuracy = 1.0\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>P_setosa</th>\n",
       "      <th>P_versicolor</th>\n",
       "      <th>P_virginica</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>setosa</th>\n",
       "      <td>16.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>versicolor</th>\n",
       "      <td>0.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>virginica</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>18.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "            P_setosa  P_versicolor  P_virginica\n",
       "setosa          16.0           0.0          0.0\n",
       "versicolor       0.0          11.0          0.0\n",
       "virginica        0.0           0.0         18.0"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Acc=accuracy(y_T,y_p1)\n",
    "print(\"accuracy =\",round(Acc,3)) \n",
    "cm=conf_matrix(y_T,y_p1)\n",
    "conf_report(cm,['setosa','versicolor','virginica'],['P_setosa','P_versicolor','P_virginica'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "accuracy = 1.0\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>P_setosa</th>\n",
       "      <th>P_versicolor</th>\n",
       "      <th>P_virginica</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>setosa</th>\n",
       "      <td>16.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>versicolor</th>\n",
       "      <td>0.0</td>\n",
       "      <td>11.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>virginica</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>18.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "            P_setosa  P_versicolor  P_virginica\n",
       "setosa          16.0           0.0          0.0\n",
       "versicolor       0.0          11.0          0.0\n",
       "virginica        0.0           0.0         18.0"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Acc=accuracy(y_T,y_p1)\n",
    "print(\"accuracy =\",round(Acc,3)) \n",
    "cm=conf_matrix(y_T,y_p1)\n",
    "conf_report(cm,['setosa','versicolor','virginica'],['P_setosa','P_versicolor','P_virginica'])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# k-fold"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "id": "UzKAJwR2ViSj"
   },
   "outputs": [],
   "source": [
    "def k_fold(Clf,dataset,k):\n",
    "    f_len=round(len(dataset)/k) \n",
    "    S=0\n",
    "    sqr=0\n",
    "    for i in range(k):\n",
    "        a=i*f_len\n",
    "        b=(i+1)*f_len\n",
    "        f_test=dataset[a:b]\n",
    "        f_train=[*dataset[:a],*dataset[b:]]\n",
    "        y_P= Clf(f_train, f_test)\n",
    "        y_T=np.array(f_test[:,-1]).astype(int)  \n",
    "        Acc=accuracy(y_T,y_P)\n",
    "        print(\"fold\",i+1,\": Acc=\",round(Acc,3)) \n",
    "        S += Acc/k\n",
    "        sqr += (Acc**2)/k  \n",
    "    mean= S\n",
    "    var=sqr-(S**2) \n",
    "    return mean,var"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {
    "id": "FMhpQXdq6oUX"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "fold 1 : Acc= 1.0\n",
      "fold 2 : Acc= 0.933\n",
      "fold 3 : Acc= 0.9\n",
      "fold 4 : Acc= 1.0\n",
      "fold 5 : Acc= 0.967\n",
      "accuracy_ave 0.96\n",
      "accuracy_var 0.0015111111111112407\n"
     ]
    }
   ],
   "source": [
    "accuracy_ave,accuracy_var=k_fold(naive_bayes,dataset,5)\n",
    "print(\"accuracy_ave\",accuracy_ave)\n",
    "print(\"accuracy_var\",accuracy_var)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "fold 1 : Acc= 0.967\n",
      "fold 2 : Acc= 0.967\n",
      "fold 3 : Acc= 0.9\n",
      "fold 4 : Acc= 1.0\n",
      "fold 5 : Acc= 1.0\n",
      "accuracy_ave 0.9666666666666666\n",
      "accuracy_var 0.0013333333333336306\n"
     ]
    }
   ],
   "source": [
    "accuracy_ave,accuracy_var=k_fold(GNB,dataset,5)\n",
    "print(\"accuracy_ave\",accuracy_ave)\n",
    "print(\"accuracy_var\",accuracy_var)"
   ]
  }
 ],
 "metadata": {
  "anaconda-cloud": {},
  "colab": {
   "collapsed_sections": [],
   "name": "practic_yas_5 (1).ipynb",
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}
