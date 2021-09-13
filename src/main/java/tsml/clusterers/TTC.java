/*
 * This file is part of the UEA Time Series Machine Learning (TSML) toolbox.
 *
 * The UEA TSML toolbox is free software: you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * The UEA TSML toolbox is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with the UEA TSML toolbox. If not, see <https://www.gnu.org/licenses/>.
 */

package tsml.clusterers;

import experiments.data.DatasetLoading;
import machine_learning.clusterers.CAST;
import machine_learning.clusterers.PAM;
import tsml.classifiers.legacy.elastic_ensemble.distance_functions.DTW;
import weka.clusterers.NumberOfClustersRequestable;
import weka.core.DenseInstance;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Arrays;

import static utilities.ClusteringUtilities.*;
import static utilities.InstanceTools.deleteClassAttribute;

/**
 * Class for a dictionary based clustering algorithm, very experimental.
 *
 * @author Matthew Middlehurst
 */
public class TTC extends EnhancedAbstractClusterer implements NumberOfClustersRequestable {

    //Aghabozorgi, Saeed, et al.
    //"A hybrid algorithm for clustering of time series data based on affinity search technique."
    //The Scientific World Journal 2014 (2014).

    private double affinityThreshold = 0.01;
    private int k = 2;

    private double[][] distanceMatrix;
    private ArrayList<Integer>[] subclusters;

    private PAM pam;

    public TTC() {
    }

    @Override
    public int numberOfClusters() throws Exception {
        return k;
    }

    @Override
    public void setNumClusters(int numClusters) throws Exception {
        k = numClusters;
    }

    @Override
    public void buildClusterer(Instances data) throws Exception {
        super.buildClusterer(data);

        zNormalise(train);

        EuclideanDistance ed = new EuclideanDistance();
        ed.setDontNormalize(true);
        distanceMatrix = createDistanceMatrix(train, ed);

        //Cluster using the CAST algorithm
        CAST cast = new CAST(distanceMatrix);
        cast.setAffinityThreshold(affinityThreshold);
        cast.buildClusterer(train);
        subclusters = cast.getClusters();
        ArrayList<double[]> affinities = cast.getClusterAffinities();

        double[][] prototypes = new double[subclusters.length][train.numAttributes()];

        //Take average of each cluster
        for (int i = 0; i < subclusters.length; i++) {
            for (int n = 0; n < train.numAttributes(); n++) {
                for (int g = 0; g < subclusters[i].size(); g++) {
                    prototypes[i][n] += train.get(subclusters[i].get(g)).value(n) * (1 - affinities.get(i)[g]);
                }

                prototypes[i][n] /= subclusters[i].size();
            }
        }

        Instances cl = new Instances(train, subclusters.length);
        for (int i = 0; i < subclusters.length; i++) {
            cl.add(new DenseInstance(1, prototypes[i]));
        }

        //Use PAM using DTW distance to cluster discretised data
        pam = new PAM();
        pam.setDistanceFunction(new DTW());
        pam.setNumClusters(k);
        pam.setNormaliseData(false);
        pam.setCopyInstances(false);
        if (seedClusterer)
            pam.setSeed(seed);
        pam.buildClusterer(cl);

        ArrayList<Integer>[] ptClusters = pam.getClusters();
        assignments = new double[train.size()];

        //Assign each instance to the cluster assigned it its subcluster using PAM
        for (int i = 1; i < k; i++) {
            for (int n = 0; n < ptClusters[i].size(); n++) {
                ArrayList<Integer> subcluster = subclusters[ptClusters[i].get(n)];

                for (int g = 0; g < subcluster.size(); g++) {
                    assignments[subcluster.get(g)] = i;
                }
            }
        }

        clusters = new ArrayList[k];

        for (int i = 0; i < k; i++) {
            clusters[i] = new ArrayList();
        }

        for (int i = 0; i < train.size(); i++) {
            clusters[(int) assignments[i]].add(i);
        }
    }

    @Override
    public int clusterInstance(Instance inst) throws Exception {
        Instance newInst = copyInstances ? new DenseInstance(inst) : inst;
        deleteClassAttribute(newInst);
        zNormalise(newInst);
        return pam.clusterInstance(newInst);
    }

    public static void main(String[] args) throws Exception {
        String dataset = "Trace";
        Instances inst = DatasetLoading.loadDataNullable("Z:\\ArchiveData\\Univariate_arff\\" + dataset + "/" +
                dataset + "_TRAIN.arff");
        Instances inst2 = DatasetLoading.loadDataNullable("Z:\\ArchiveData\\Univariate_arff\\" + dataset + "/" +
                dataset + "_TEST.arff");
        inst.setClassIndex(inst.numAttributes() - 1);
        inst.addAll(inst2);

        TTC k = new TTC();
        k.setSeed(0);
        k.k = inst.numClasses();
        k.buildClusterer(inst);

        System.out.println(k.clusters.length);
        System.out.println(Arrays.toString(k.clusters));
        System.out.println(randIndex(k.assignments, inst));
    }
}
