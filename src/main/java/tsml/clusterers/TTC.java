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
package tsml.clusterers;

import experiments.data.DatasetLoading;
import machine_learning.clusterers.CAST;
import machine_learning.clusterers.PAM;
import tsml.classifiers.legacy.elastic_ensemble.distance_functions.DTW;
import weka.core.DenseInstance;
import weka.core.EuclideanDistance;
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
public class TTC extends AbstractTimeSeriesClusterer {

    //Aghabozorgi, Saeed, et al.
    //"A hybrid algorithm for clustering of time series data based on affinity search technique."
    //The Scientific World Journal 2014 (2014).

    private double affinityThreshold = 0.01;
    private int k = 2;
    private int seed = Integer.MIN_VALUE;

    private double[][] distanceMatrix;
    private ArrayList<Integer>[] subclusters;

    public TTC(){}

    @Override
    public int numberOfClusters() throws Exception { return k; }

    public void setNumberOfClusters(int n){ k = n; }

    public void setSeed(int seed){ this.seed = seed; }

    @Override
    public void buildClusterer(Instances data) throws Exception {
        if (copyInstances){
            data = new Instances(data);
        }

        deleteClassAttribute(data);
        zNormalise(data);

        EuclideanDistance ed = new EuclideanDistance();
        ed.setDontNormalize(true);
        distanceMatrix = createDistanceMatrix(data, ed);

        //Cluster using the CAST algorithm
        CAST cast = new CAST(distanceMatrix);
        cast.setAffinityThreshold(affinityThreshold);
        cast.buildClusterer(data);
        subclusters = cast.getClusters();
        ArrayList<double[]> affinities = cast.getClusterAffinities();

        double[][] prototypes = new double[subclusters.length][data.numAttributes()];

        //Take average of each cluster
        for (int i = 0; i < subclusters.length; i++){
            for (int n = 0; n < data.numAttributes(); n++){
                for (int g = 0; g < subclusters[i].size(); g++){
                    prototypes[i][n] += data.get(subclusters[i].get(g)).value(n) * (1-affinities.get(i)[g]);
                }

                prototypes[i][n] /= subclusters[i].size();
            }
        }

        Instances cl = new Instances(data, subclusters.length);
        for (int i = 0; i < subclusters.length; i++){
            cl.add(new DenseInstance(1, prototypes[i]));
        }


        //Use PAM using DTW distance to cluster discretised data
        PAM pam = new PAM();
        pam.setDistanceFunction(new DTW());
        pam.setNumberOfClusters(k);
        pam.setSeed(seed);
        pam.buildClusterer(cl);

        ArrayList<Integer>[] ptClusters = pam.getClusters();
        assignments = new int[data.size()];

        //Assign each instance to the cluster assigned it its subcluster using PAM
        for (int i = 1; i < k; i++){
            for (int n = 0; n < ptClusters[i].size(); n++){
                ArrayList<Integer> subcluster = subclusters[ptClusters[i].get(n)];

                for (int g = 0; g < subcluster.size(); g++){
                    assignments[subcluster.get(g)] = i;
                }
            }
        }

        clusters = new ArrayList[k];

        for (int i = 0; i < k; i++){
            clusters[i] = new ArrayList();
        }

        for (int i = 0; i < data.size(); i++){
            for (int n = 0; n < k; n++){
                if(n == assignments[i]){
                    clusters[n].add(i);
                    break;
                }
            }
        }
    }

    public static void main(String[] args) throws Exception{
        String dataset = "Trace";
        Instances inst = DatasetLoading.loadDataNullable("Z:\\ArchiveData\\Univariate_arff\\" + dataset + "/" +
                dataset + "_TRAIN.arff");
        Instances inst2 = DatasetLoading.loadDataNullable("Z:\\ArchiveData\\Univariate_arff\\" + dataset + "/" +
                dataset + "_TEST.arff");
        inst.setClassIndex(inst.numAttributes()-1);
        inst.addAll(inst2);

        TTC k = new TTC();
        k.seed = 0;
        k.k = inst.numClasses();
        k.buildClusterer(inst);

        System.out.println(k.clusters.length);
        System.out.println(Arrays.toString(k.clusters));
        System.out.println(randIndex(k.assignments, inst));
    }
}
