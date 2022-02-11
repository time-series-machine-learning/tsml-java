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


import machine_learning.clusterers.CAST;
import machine_learning.clusterers.DensityPeaks;
import machine_learning.clusterers.KMeans;
import machine_learning.clusterers.KMedoids;
import tsml.clusterers.KShape;
import tsml.clusterers.TTC;
import tsml.clusterers.UnsupervisedShapelets;
import weka.clusterers.Clusterer;
import weka.clusterers.EM;
import weka.clusterers.NumberOfClustersRequestable;
import weka.clusterers.SimpleKMeans;

import java.util.Arrays;
import java.util.HashSet;

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
        String cls = exp.estimatorName.toLowerCase();

        Clusterer c = null;
        switch(cls) {
            case "simplekmeans":
                c = new SimpleKMeans();
                break;
            case "em":
                c = new EM();
                break;

            case "kmeans":
                c = new KMeans();
                break;
            case "kmedoids":
                c = new KMedoids();
                break;
            case "densitypeaks":
                c = new DensityPeaks();
                break;
            case "cast":
                c = new CAST();
                break;

            case "ushapelets":
                c = new UnsupervisedShapelets();
                break;
            case "kshape":
                c = new KShape();
                break;
            case "ttc":
                c = new TTC();
                break;

            case "ushapelets2":
                c = new UnsupervisedShapelets();
                ((UnsupervisedShapelets)c).setExhaustiveSearch(true);
                break;
            case "ushapelets3":
                c = new UnsupervisedShapelets();
                ((UnsupervisedShapelets)c).setRandomSearchProportion(0.2);
                break;
            case "ushapelets4":
                c = new UnsupervisedShapelets();
                ((UnsupervisedShapelets)c).setUseKMeans(false);
                break;
            case "ushapelets5":
                c = new UnsupervisedShapelets();
                ((UnsupervisedShapelets)c).setExhaustiveSearch(true);
                ((UnsupervisedShapelets)c).setUseKMeans(false);
                break;
            case "ushapelets6":
                c = new UnsupervisedShapelets();
                ((UnsupervisedShapelets)c).setRandomSearchProportion(0.2);
                ((UnsupervisedShapelets)c).setUseKMeans(false);
                break;

            default:
                System.out.println("Unknown clusterer " + cls);
        }

        if (c instanceof NumberOfClustersRequestable)
            ((NumberOfClustersRequestable)c).setNumClusters(exp.numClassValues);

        return c;
    }


    /**
     * This method redproduces the old usage exactly as it was in old experiments.java.
     * If you try build any clusterer that uses any experimental info other than
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
