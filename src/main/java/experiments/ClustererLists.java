/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package experiments;


import machine_learning.clusterers.KMeans;
import weka.classifiers.Classifier;
import weka.clusterers.*;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Locale;

/**
 *
 * @author James Large (james.large@uea.ac.uk) and Tony Bagnall
 */
public class ClustererLists {

    //All implemented clusterers in tsml
    //<editor-fold defaultstate="collapsed" desc="All univariate time series classifiers">
    public static String[] allClst={
         "KMeans",
         "KShape"
    };
    //</editor-fold>
    public static HashSet<String> allClusterers=new HashSet<>( Arrays.asList(allClst));


    /**
     *
     * setClusterer, which takes the experimental
     * arguments themselves and therefore the classifiers can take from them whatever they
     * need, e.g the dataset name, the fold id, separate checkpoint paths, etc.
     *
     * To take this idea further, to be honest each of the TSC-specific classifiers
     * could/should have a constructor and/or factory that builds the classifier
     * from the experimental args.
     *
     * previous usage was setClusterer(String clusterer name, int fold).
     * this can be reproduced with setClassifierClassic below.
     *
     */
    public static Clusterer setClusterer(ExperimentalArguments exp) throws Exception {
        String cls=exp.estimatorName.toLowerCase();
        Clusterer c = null;
        switch(cls) {
            case "kmeans":
                c = new SimpleKMeans();
                break;
/*            case "dbscan":
                c = new DBScan();
                break;
*/          case "em":
                c = new EM();
                break;
        }
        if(c instanceof NumberOfClustersRequestable)
            ((NumberOfClustersRequestable)c).setNumClusters(exp.numClassValues);
        return c;
    }


    /**
     * This method redproduces the old usage exactly as it was in old experiments.java.
     * If you try build any classifier that uses any experimental info other than
     * expestimatorName or exp.foldID, an exception will be thrown.
     * @param clusterer
     * @param fold
     * @return clusterer
     */
    public static Clusterer setClustererClassic(String clusterer, int fold) throws Exception {
        ExperimentalArguments exp=new ExperimentalArguments();
        exp.estimatorName =clusterer;
        exp.foldId=fold;
        return setClusterer(exp);
    }


    public static void main(String[] args) throws Exception {

    }
}
