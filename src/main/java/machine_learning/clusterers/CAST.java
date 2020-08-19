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
package machine_learning.clusterers;

import experiments.data.DatasetLoading;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Arrays;

import static utilities.ClusteringUtilities.createDistanceMatrix;
import static utilities.InstanceTools.deleteClassAttribute;
import static utilities.Utilities.maxIndex;
import static utilities.Utilities.minIndex;

/**
 * Implementation of the CAST clustering algorithm.
 *
 * @author Matthew Middlehurst
 */
public class CAST extends AbstractVectorClusterer {

    //Ben-Dor, Amir, Ron Shamir, and Zohar Yakhini.
    //"Clustering gene expression patterns."
    //Journal of computational biology 6.3-4 (1999): 281-297.

    private double affinityThreshold = 0.1;
    private boolean dynamicAffinityThreshold = false;
    private double eCastThreshold = 0.25;

    private double[][] distanceMatrix;
    private boolean hasDistances = false;

    private ArrayList<double[]> clusterAffinities;

    public CAST(){}

    public CAST(double[][] distanceMatrix){
        this.distanceMatrix = distanceMatrix;
        this.hasDistances = true;
    }

    @Override
    public int numberOfClusters() {
        return clusters.length;
    }

    public ArrayList<double[]> getClusterAffinities(){
        return clusterAffinities;
    }

    public void setAffinityThreshold(double d){
        affinityThreshold = d;
    }

    public void setDynamicAffinityThreshold(boolean b){
        dynamicAffinityThreshold = b;
    }

    @Override
    public void buildClusterer(Instances data) throws Exception {
        if (copyInstances){
            data = new Instances(data);
        }

        deleteClassAttribute(data);

        if (normaliseData){
            normaliseData(data);
        }

        if(!hasDistances){
            distanceMatrix = createDistanceMatrix(data, distFunc);
        }

        normaliseDistanceMatrix();

        //Main CAST loop
        ArrayList<ArrayList<Integer>> subclusters = runCAST();

        //Create and store an ArrayList for each cluster containing indexes of
        //points inside the cluster
        assignments = new int[data.size()];
        clusters = new ArrayList[subclusters.size()];

        for (int i = 0; i < subclusters.size(); i++){
            clusters[i] = new ArrayList();

            for (int n = 0; n < subclusters.get(i).size(); n++){
                clusters[i].add(subclusters.get(i).get(n));
                assignments[subclusters.get(i).get(n)] = i;
            }
        }
    }

    private ArrayList<ArrayList<Integer>> runCAST(){
        ArrayList<ArrayList<Integer>> subclusters = new ArrayList();
        ArrayList<Integer> indicies = new ArrayList(distanceMatrix.length);
        for (int i = 0; i < distanceMatrix.length; i++){
            indicies.add(i);
        }
        clusterAffinities = new ArrayList();

        double[] subclusterAffinities = null;

        while (indicies.size() > 0){
            ArrayList<Integer> subcluster = new ArrayList();
            boolean change = true;

            //E-cast
            if (dynamicAffinityThreshold){
                computeThreshold(indicies);
            }

            subcluster.add(indicies.remove(initialiseCluster(indicies)));

            //While changes still happen continue to add and remove items from the cluster.
            while(change){
                change = false;
                double[] indiciesAffinities = getAffinities(indicies, subcluster);
                int minIdx = minIndex(indiciesAffinities);

                //Addition step
                while (indiciesAffinities.length > 0 && indiciesAffinities[minIdx] <=
                        affinityThreshold*subcluster.size()){
                    subcluster.add(indicies.remove(minIdx));
                    indiciesAffinities = getAffinities(indicies, subcluster);
                    minIdx = minIndex(indiciesAffinities);

                    if(!change){
                        change = true;
                    }
                }

                subclusterAffinities = getAffinities(subcluster, subcluster);
                int maxIdx = maxIndex(subclusterAffinities);

                //Removal step
                while (subclusterAffinities[maxIdx] > affinityThreshold*(subcluster.size()-1)){
                    indicies.add(subcluster.remove(maxIdx));
                    subclusterAffinities = getAffinities(subcluster, subcluster);
                    maxIdx = maxIndex(subclusterAffinities);

                    if(!change){
                        change = true;
                    }
                }
            }

            //Add the cluster and the affinities of each member to itself, items in subcluster removed from indicies
            //pool
            clusterAffinities.add(subclusterAffinities);
            subclusters.add(subcluster);
        }

        return subclusters;
    }

    private double[] getAffinities(ArrayList<Integer> indicies, ArrayList<Integer> subcluster){
        double[] affinities = new double[indicies.size()];

        for (int n = 0; n < affinities.length; n++) {
            for (int i = 0; i < subcluster.size(); i++) {
                if (indicies.get(n).equals(subcluster.get(i))) continue;

                if (indicies.get(n) > subcluster.get(i)) {
                    affinities[n] += distanceMatrix[indicies.get(n)][subcluster.get(i)];
                } else {
                    affinities[n] += distanceMatrix[subcluster.get(i)][indicies.get(n)];
                }
            }
        }

        return affinities;
    }

    public int initialiseCluster(ArrayList<Integer> indicies){
        double minDist = Double.MAX_VALUE;
        int minIdx = 0;

        for (int n = 0; n < indicies.size(); n++) {
            for (int i = 0; i < indicies.size(); i++) {
                if(indicies.get(n).equals(indicies.get(i))) continue;
                double dist;

                if (indicies.get(n) > indicies.get(i)) {
                    dist = distanceMatrix[indicies.get(n)][indicies.get(i)];
                } else {
                    dist = distanceMatrix[indicies.get(i)][indicies.get(n)];
                }

                if (dist < minDist){
                    minDist = dist;
                    minIdx = n;
                }
            }
        }

        return minIdx;
    }

    private void normaliseDistanceMatrix(){
        double maxDist = Double.MIN_VALUE;
        double minDist = Double.MAX_VALUE;

        for (int i = 0; i < distanceMatrix.length; i++){
            for (int n = 0; n < i; n++){
                if (distanceMatrix[i][n] > maxDist){
                    maxDist = distanceMatrix[i][n];
                }
                if (distanceMatrix[i][n] < minDist){
                    minDist = distanceMatrix[i][n];
                }
            }
        }

        for (int i = 0; i < distanceMatrix.length; i++){
            for (int n = 0; n < i; n++){
                distanceMatrix[i][n] = (distanceMatrix[i][n] - minDist)/(maxDist - minDist);
            }
        }
    }

    //Bellaachia, Abdelghani, et al.
    //"E-CAST: a data mining algorithm for gene expression data."
    //Proceedings of the 2nd International Conference on Data Mining in Bioinformatics. Springer-Verlag, 2002.

    private void computeThreshold(ArrayList<Integer> indicies){
        double a = 0;
        int count = 0;

        for (int n = 0; n < indicies.size(); n++) {
            for (int i = 0; i < indicies.size(); i++) {
                if (indicies.get(n).equals(indicies.get(i))) continue;
                double dist;

                if (indicies.get(n) > indicies.get(i)) {
                    dist = distanceMatrix[indicies.get(n)][indicies.get(i)];
                } else {
                    dist = distanceMatrix[indicies.get(i)][indicies.get(n)];
                }

                if (dist < eCastThreshold){
                    a += dist - eCastThreshold;
                    count++;
                }
            }
        }

        affinityThreshold = (a/count)+eCastThreshold;
        if (Double.isNaN(affinityThreshold)) affinityThreshold = eCastThreshold;
    }

    public static void main(String[] args) throws Exception{
        String[] datasets = {"Z:\\Data Working Area\\ClusteringTestDatasets\\DensityPeakVector\\aggregation.arff",
                "Z:\\Data Working Area\\ClusteringTestDatasets\\DensityPeakVector\\clustersynth.arff",
                "Z:\\Data Working Area\\ClusteringTestDatasets\\DensityPeakVector\\dptest1k.arff",
                "Z:\\Data Working Area\\ClusteringTestDatasets\\DensityPeakVector\\dptest4k.arff",
                "Z:\\Data Working Area\\ClusteringTestDatasets\\DensityPeakVector\\flame.arff",
                "Z:\\Data Working Area\\ClusteringTestDatasets\\DensityPeakVector\\spiral.arff"};
        String[] names = {"aggre", "synth", "dptest1k", "dptest4k", "flame", "spiral"};
        boolean output = true;

        if (output){
            System.out.println("cd('Z:\\Data Working Area\\ClusteringTestDatasets\\DensityPeakVector\\" +
                    "DensityPeakVector')");
            System.out.println("load('matlabCluster.mat')");
            System.out.println("k = [1,2,3,4,5,6,7,8,9,10]");
        }

        for (int i = 0; i < datasets.length; i++){
            Instances inst = DatasetLoading.loadDataNullable(datasets[i]);
            inst.setClassIndex(inst.numAttributes()-1);
            CAST cast = new CAST();
            cast.setDynamicAffinityThreshold(true);
            cast.buildClusterer(inst);

            if(output){
                System.out.println(names[i] + "c = " + Arrays.toString(cast.assignments));
                System.out.println("figure");
                System.out.println("scatter(" + names[i] + "x," + names[i] + "y,[],scatterColours(" + names[i] + "c))");
            }
        }
    }
}
