import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import streamlit as st

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('lessedited.csv')
data = data.drop('Unnamed: 0', axis=1)
data["Round"] = data["Round"].replace(["Finals", "Quarter-Finals", "Semi-Finals"],
                                      ["The Final", "Semifinals", "Quarterfinals"])
data["Date"] = pd.to_datetime(data["Date"], format="mixed")

df = pd.read_csv("tennis13-22.csv").drop('Unnamed: 0', axis=1)
df["Date"] = pd.to_datetime(df["Date"], format="mixed")


def getChance(dat, play, op, tour, prank, orank):
    dat["PlayW"] = 0
    dat["OppW"] = 0
    dat.loc[dat["Winner"] == play, "PlayW"] = 1
    dat.loc[dat["Loser"] == play, "PlayW"] = -1
    dat.loc[dat["Winner"] == op, "OppW"] = 1
    dat.loc[dat["Loser"] == op, "OppW"] = -1

    return dat


def mod_standard_score(x):
    x_mss = []
    median = np.median(x)
    absdiff = np.abs(x - median)
    aad = (np.median(absdiff) / (len(x)))
    for i in x:
        x_mss.append(((i - median) / aad) / len(x))
    return x_mss


def calculate_regression_goodness_of_fit(ys, y_hat):
    ss_total = 0
    ss_residual = 0
    ss_regression = 0
    y_mean = ys.mean()
    for i in range(len(ys)):
        ss_total += np.square(ys[i] - y_mean)
        ss_residual += np.square(ys[i] - y_hat[i])
        ss_regression += np.square(y_hat[i] - y_mean)
    r_square = ss_regression / ss_total
    rmse = np.sqrt(ss_residual / float(len(ys)))
    return r_square[0], rmse[0]


def rsqRMSEcalc(ex, wy, ran=-1):
    X = ex.values
    X = np.reshape(X, (len(ex), 1))
    y = wy.values
    y = np.reshape(y, (len(wy), 1))

    knn = neighbors.KNeighborsRegressor(n_neighbors=(round(len(y) ** 0.5) + 1), weights='uniform')
    mod = knn.fit(X, y)
    x = np.reshape(ex.values, (len(ex), 1)) + 0.0001
    y_hat = mod.predict(x)
    calculate_regression_goodness_of_fit(y, y_hat)

    rsquare_arr = []
    rmse_arr = []
    rangeUpper = ran
    if (rangeUpper < 1):
        rangeUpper = round((len(y) ** 0.5) / 5)

    for k in range(2, rangeUpper):
        knn = neighbors.KNeighborsRegressor(n_neighbors=k)
        y_hat = knn.fit(X, y).predict(x)
        rsquare, rmse = calculate_regression_goodness_of_fit(y, y_hat)
        rmse_arr.append(rmse)
        rsquare_arr.append(rsquare)
    return rsquare_arr, rmse_arr


def rsqRMSEcalc2D(ex, wy, ran=-1):
    X = ex.values
    X = np.reshape(X, (len(ex), len(ex.iloc[1])))
    y = wy.values
    y = np.reshape(y, (len(wy), 1))

    knn = neighbors.KNeighborsRegressor(n_neighbors=(round(len(y) ** 0.5) + 1), weights='uniform')
    mod = knn.fit(X, y)
    x = np.reshape(ex.values, (len(ex), len(ex.iloc[1]))) + 0.0001
    y_hat = mod.predict(x)
    calculate_regression_goodness_of_fit(y, y_hat)

    rsquare_arr = []
    rmse_arr = []
    rangeUpper = ran
    if (rangeUpper < 1):
        rangeUpper = round((len(y) ** 0.5) / 5)

    for k in range(2, rangeUpper):
        knn = neighbors.KNeighborsRegressor(n_neighbors=k)
        y_hat = knn.fit(X, y).predict(x)
        rsquare, rmse = calculate_regression_goodness_of_fit(y, y_hat)
        rmse_arr.append(rmse)
        rsquare_arr.append(rsquare)
    return rsquare_arr, rmse_arr


def plotRSQ(rsq, rms, rup, ti1="RSquare for the KNeighborsRegressor", ti2="RMSE for the KNeighborsRegressor"):
    plt.subplot(2, 1, 1)
    plt.plot(range(2, rup), rsq, c='k', label='R Square')
    plt.axis('tight')
    plt.xlabel('k number of nearest neighbours')
    plt.ylabel('RSquare')
    plt.legend(loc='upper right')
    plt.title(ti1)
    plt.subplots_adjust(hspace=0.5)

    plt.subplot(2, 1, 2)
    plt.plot(range(2, rup), rms, c='k', label='RMSE')
    plt.axis('tight')
    plt.xlabel('k number of nearest neighbours')
    plt.ylabel('RMSE')
    plt.legend(loc='upper left')
    plt.title(ti2)
    plt.show()


def plotPredict(ex, wy, nn, lins, x=0):
    X = ex.values
    X = np.reshape(X, (len(ex), 1))
    y = wy.values
    y = np.reshape(y, (len(wy), 1))

    n_neighbors = nn
    if x == 0:
        x = np.linspace(X.min(), X.max(), lins)[:, np.newaxis]
    plt.clf()
    for i, weights in enumerate(['uniform', 'distance']):
        knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
        y_hat = knn.fit(X, y).predict(x)

        plt.subplot(2, 1, i + 1)
        plt.scatter(X, y, c='k', label='data')
        plt.plot(x, y_hat, c='b', label='prediction')
        plt.axis('tight')
        plt.xlabel(ex.name)
        plt.ylabel(wy.name)
        plt.legend(loc='upper left')
        plt.title("KNeighborsRegressor (k = %i, weights = '%s')" % (n_neighbors, weights))
        plt.subplots_adjust(hspace=0.5)

    st.pyplot(plt)


def plotPred2D(ex, wy, nn, lins, x=0):
    X = ex.values
    X = np.reshape(X, (len(ex), len(ex.iloc[1])))
    y = wy.values
    y = np.reshape(y, (len(wy), 1))

    if (x == 0):
        x = np.mgrid[X.min():X.max():complex(lins), X.min():X.max():complex(lins)].reshape(2, -1).T

    for i, weights in enumerate(['uniform', 'distance']):
        knn = neighbors.KNeighborsRegressor(nn, weights=weights)
        y_hat = knn.fit(X, y).predict(x)

        plt.subplot(2, 1, i + 1)
        plt.scatter(X, y, c='k', label='data')
        plt.plot(x, y_hat, c='b', label='prediction')
        plt.axis('tight')
        plt.xlabel(ex.name)
        plt.ylabel(wy.name)
        plt.legend(loc='upper left')
        plt.title("KNeighborsRegressor (k = %i, weights = '%s')" % (nn, weights))
        plt.subplots_adjust(hspace=0.5)

    plt.show()


def testWin(p1, p2, dat, playdat, oppdat, rm1, rm2):
    dftest = pd.DataFrame(columns=["WRank", "LRank"])
    dftest.loc[0] = [p1, p2]
    testW = dftest["WRank"][0]
    testL = dftest["LRank"][0]

    if dat[dat["WRank"] == dftest["WRank"][0]].empty:
        testW = dat.iloc[(dat['WRank'] - testW).abs().argsort()[:1]]["AdjWRank"].iloc[0]
    else:
        testW = dat[dat["WRank"] == testW]["AdjWRank"].iloc[0]
    if dat[dat["LRank"] == dftest["LRank"][0]].empty:
        testL = dat.iloc[(dat['LRank'] - testL).abs().argsort()[:1]]["LRank"].iloc[0]
    else:
        testL = dat[dat["LRank"] == testL]["AdjLRank"].iloc[0]

    dftest["AdjWRank"] = testW
    dftest["AdjLRank"] = testL
    dft2 = pd.DataFrame(columns=["AdjWRank", "AdjLRank"])
    dft2.loc[0] = [testL, testW]
    knn = neighbors.KNeighborsRegressor(rm1.index(min(rm1)) + 2, weights="uniform")
    a = knn.fit(playdat[["AdjWRank", "AdjLRank"]], playdat["PlayW"]).predict(dftest[["AdjWRank", "AdjLRank"]])
    knn2 = neighbors.KNeighborsRegressor(rm2.index(min(rm2)) + 2, weights="uniform")
    b2 = knn2.fit(oppdat[["AdjWRank", "AdjLRank"]], oppdat["OppW"]).predict(dft2[["AdjWRank", "AdjLRank"]])
    return a, b2


def getAccuracy(xval, yval, rms, size):
    X_train, X_test, y_train, y_test = train_test_split(xval, yval, random_state=1, test_size=size)
    nb = GaussianNB()
    treeclf = DecisionTreeClassifier(max_depth=5, random_state=1)
    knn_classifier = KNeighborsClassifier(n_neighbors=rms.index(min(rms)) + 2, metric='euclidean')
    return nb.fit(X_train, y_train).score(X_test, y_test), treeclf.fit(X_train, y_train).score(X_test,
                                                                                               y_test), knn_classifier.fit(
        X_train, y_train).score(X_test, y_test)


# sg.theme('Dark')
playerName = st.text_input("Player Name (e.g. Nadal R.):", "Nadal R.", key="as")
oppName = st.text_input("Opponent Name (e.g. Djokovic N.):", "Djokovic N.", key="21")
tour = st.text_input("[OPTIONAL] Tournament:", "", key="rftj")
playrank = st.text_input("Player's ATP ranking:", "3", key="aluy")
opprank = st.text_input("Opponent's ATP ranking:", "5", key="51")

if not playerName or not oppName:
    st.error("Error: Player or Opponent not given.")
elif playrank.isdigit() is False or opprank.isdigit() is False:
    st.error("Error: given ranks are invalid (not numerical).")
elif int(playrank) <= 0 or int(opprank) <= 0:
    st.error("Error: given ranks are invalid (less than or equal to 0)")
elif playerName == oppName:
    st.error("Error: Player and Opponent are the same.")
elif len(df[(df['Winner'] == playerName) | (df["Loser"] == playerName)]) <= 0:
    st.error("Error: Player Name is not present in data set.")
elif len(df[(df['Winner'] == oppName) | (df["Loser"] == oppName)]) <= 0:
    st.error("Error: Opponent Name is not present in data set.")
else:
    playrank = int(playrank)
    opprank = int(opprank)
    nudf = df[((df['Winner'] == playerName) | (df["Loser"] == playerName)) | (
            (df['Winner'] == oppName) | (df["Loser"] == oppName))]
    nudf = getChance(nudf, playerName, oppName, tour, playrank, opprank)
    nudf["AdjWRank"] = mod_standard_score(nudf["WRank"])
    nudf["AdjLRank"] = mod_standard_score(nudf["LRank"])
    dfplay = nudf[(nudf['Winner'] == playerName) | (nudf["Loser"] == playerName)]
    dfopp = nudf[(nudf['Winner'] == oppName) | (nudf["Loser"] == oppName)]
    dfboth = nudf[((nudf['Winner'] == playerName) | (nudf["Loser"] == playerName)) & (
            (nudf['Winner'] == oppName) | (nudf["Loser"] == oppName))]
    if tour:
        if len(dfplay[dfplay["Tournament"] == tour]) > 0 & len(dfopp[dfopp["Tournament"] == tour]) > 0:
            dtest = dfplay[dfplay["Tournament"] == tour]
            rtest, rmsetest = rsqRMSEcalc2D(dtest[["AdjWRank", "AdjLRank"]], dtest["PlayW"], min(100, len(dtest)))
            if len(rmsetest) > 0:
                dtest = dfopp[dfopp["Tournament"] == tour]
                rtest, rmsetest = rsqRMSEcalc2D(dtest[["AdjWRank", "AdjLRank"]], dtest["PlayW"], min(100, len(dtest)))
                if len(rmsetest) > 0:
                    dfplay = dfplay[dfplay["Tournament"] == tour]
                    dfopp = dfopp[dfopp["Tournament"] == tour]

    maxnplay = min(100, len(dfplay))
    maxnopp = min(100, len(dfopp))
    maxnboth = min(100, len(dfboth))
    rsquare1, rmse1 = rsqRMSEcalc2D(dfplay[["AdjWRank", "AdjLRank"]], dfplay["PlayW"], maxnplay)
    if len(rmse1) <= 0:
        st.error("Error: Player Name does not appear in data set enough to predict.")
    rsquare2, rmse2 = rsqRMSEcalc2D(dfopp[["AdjWRank", "AdjLRank"]], dfopp["OppW"], maxnopp)
    if len(rmse2) <= 0:
        st.error("Error: Opponent Name does not appear in data set enough to predict.")
    rsquare3, rmse3 = rsqRMSEcalc2D(dfboth[["AdjWRank", "AdjLRank"]], dfboth["OppW"], maxnboth)

    p1win, p2win = testWin(playrank, opprank, nudf, dfplay, dfopp, rmse1, rmse2)
    p3win, p4win = testWin(opprank, playrank, nudf, dfopp, dfplay, rmse2, rmse1)
    n, t, k = getAccuracy(dfplay[["AdjWRank", "AdjLRank"]], dfplay["PlayW"], rmse1, 0.2)
    acc1 = n + t + (k * 1.5)
    n2, t2, k2 = getAccuracy(dfopp[["AdjWRank", "AdjLRank"]], dfopp["OppW"], rmse2, 0.2)
    acc2 = n2 + t2 + (k2 * 1.5)
    # Double-Check?
    n3, t3, k3 = getAccuracy(dfplay[["AdjWRank", "AdjLRank"]], dfplay["OppW"], rmse1, 0.2)
    acc3 = n3 + t3 + (k3 * 1.5)
    n4, t4, k4 = getAccuracy(dfopp[["AdjWRank", "AdjLRank"]], dfopp["PlayW"], rmse2, 0.2)
    acc4 = n4 + t4 + (k4 * 1.5)
    dom1, dom2 = p1win, p2win
    weak1, weak2 = p4win, p3win
    d1acc, d2acc, w1acc, w2acc = acc1, acc2, acc4, acc3
    if acc4 > acc1:
        dom1 = p4win
        d1acc = acc4
        weak1 = p1win
        w1acc = acc1
    if acc3 > acc2:
        dom2 = p3win
        d2acc = acc3
        weak2 = p2win
        w2acc = acc2

    playres = (dom1[0] * d1acc * 1.5) + (weak1[0] * w1acc)
    oppres = (dom2[0] * d2acc * 1.5) + (weak2[0] * w2acc)
    str1 = ""
    if len(rmse3) > 0:
        p1win2, p2win2 = testWin(playrank, opprank, nudf, dfboth, dfboth, rmse3, rmse3)
        n5, t5, k5 = getAccuracy(dfboth[["AdjWRank", "AdjLRank"]], dfboth["PlayW"], rmse3, 0.2)
        acc5 = n5 + t5 + (k5 * 1.5)
        playres += (p1win2[0] * acc5 * 1.5)
        oppres += (p2win2[0] * acc5 * 1.5)
        str1 = "\n\n**Accuracy of P1 VS P2 predictions**\n\nNB %.4f" % n5 + ",  Tree: %.4f" % t5 + ",  kNN: %.4f" % k5
        str2 = "%.2f" % p1win2[0] + ", %.2f" % p2win2[0]
        playres /= 3
        oppres /= 3
    else:
        str1 = "\n\nPlayer VS Opponent Only could not be calculated (too few matches between them in dataset)"
        str2 = "N/A"
        playres *= 0.5
        oppres *= 0.5

    if (playres > oppres):
        winrar = playerName
    elif (oppres > playres):
        winrar = oppName
    else:
        winrar = "Unclear"

    #if (playres < 0):
    #    oppres -= playres
    #    playres = 0
    #if (oppres < 0):
    #    playres -= oppres
    #    oppres = 0
    chanceP = (playres)
    chanceO = (oppres)
    st.write("**Estimated Winner: " + winrar + "**")
    st.write("**Est. Likelihood of winning:** %.2f" % chanceP + ", %.2f" % chanceO + " (Player, Opponent)")
    st.write("**Dominant Prediction:** %.2f" % dom1[0] + ", %.2f" % dom2[0])
    st.write("**Secondary Prediction:** %.2f" % weak1[0] + ", %.2f" % weak2[0])
    st.write("**Shared Matches Only Prediction:** " + str2)
    st.write("**Accuracy of Player Win data**")
    st.write("NB: %.4f" % n + ",  Tree: %.4f" % t + ",  kNN: %.4f" % k)
    st.write("**Accuracy of Opponent Win data**")
    st.write("NB: %.4f" % n2 + ",  Tree: %.4f" % t2 + ",  kNN: %.4f" % k2)
    st.write("**Accuracy of Player's Opponent Win Data (estimated opponent)**")
    st.write("NB: %.4f" % n3 + ",  Tree: %.4f" % t3 + ",  kNN: %.4f" % k3)
    st.write("**Accuracy of Opponent's Opponent Win Data (estimated player)**")
    st.write("NB: %.4f" % n4 + ",  Tree: %.4f" % t4 + ",  kNN: %.4f" % k4 + str1)

    st.write("**Prediction Plots for Player's Wins** (based on adj. ranking of all players battled)")
    plotPredict(dfplay["AdjWRank"], dfplay["PlayW"], rmse1.index(min(rmse1)) + 2, min(len(rmse1), 100))
    plotPredict(dfplay["AdjLRank"], dfplay["PlayW"], rmse1.index(min(rmse1)) + 2, min(len(rmse1), 100))

    st.write("**Prediction Plots for Opponent's Wins** (based on adj. ranking of all players battled)")
    plotPredict(dfopp["AdjWRank"], dfopp["OppW"], rmse2.index(min(rmse2)) + 2, min(len(rmse2), 100))
    plotPredict(dfopp["AdjLRank"], dfopp["OppW"], rmse2.index(min(rmse2)) + 2, min(len(rmse2), 100))

    if len(rmse3) > 0:
        st.write("**Prediction Plots for VS Matches** (based on adj. ranking of all players battled)")
        plotPredict(dfboth["AdjWRank"], dfboth["PlayW"], rmse3.index(min(rmse3)) + 2, len(rmse3))
        plotPredict(dfboth["AdjWRank"], dfboth["OppW"], rmse3.index(min(rmse3)) + 2, len(rmse3))
