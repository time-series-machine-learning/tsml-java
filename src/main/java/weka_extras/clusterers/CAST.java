package weka_extras.clusterers;

import weka.core.Instances;

import java.util.ArrayList;

import static utilities.ClusteringUtilities.createDistanceMatrix;
import static utilities.InstanceTools.deleteClassAttribute;
import static utilities.Utilities.maxIndex;
import static utilities.Utilities.minIndex;

public class CAST extends AbstractVectorClusterer {

    //Ben-Dor, Amir, Ron Shamir, and Zohar Yakhini.
    //"Clustering gene expression patterns."
    //Journal of computational biology 6.3-4 (1999): 281-297.

    private double affinityThreshold = 0.3;
    private boolean dynamicAffinityThreshold = false;

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

        ArrayList<ArrayList<Integer>> subclusters = runCAST();

        //Create and store an ArrayList for each cluster containing indexes of
        //points inside the cluster.
        cluster = new int[data.size()];
        clusters = new ArrayList[subclusters.size()];

        for (int i = 0; i < subclusters.size(); i++){
            clusters[i] = new ArrayList();

            for (int n = 0; n < subclusters.get(i).size(); n++){
                clusters[i].add(subclusters.get(i).get(n));
                cluster[subclusters.get(i).get(n)] = i;
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

        double[] indiciesAffinities;
        double[] subclusterAffinities = null;

        while (indicies.size() > 0){
            ArrayList<Integer> subcluster = new ArrayList();
            boolean change = true;

            if (dynamicAffinityThreshold){
                computeThreshold(indicies);
            }

            subcluster.add(indicies.remove(initialiseCluster(indicies)));

            while(change){
                change = false;
                indiciesAffinities = getAffinities(indicies, subcluster);
                int minIdx = minIndex(indiciesAffinities);

                while (indiciesAffinities.length > 0 && indiciesAffinities[minIdx] <= affinityThreshold*subcluster.size()){
                    subcluster.add(indicies.remove(minIdx));
                    indiciesAffinities = getAffinities(indicies, subcluster);
                    minIdx = minIndex(indiciesAffinities);

                    if(!change){
                        change = true;
                    }
                }

                subclusterAffinities = getAffinities(subcluster, subcluster);
                int maxIdx = maxIndex(subclusterAffinities);

                while (subclusterAffinities[maxIdx] > affinityThreshold*subcluster.size()){
                    indicies.add(subcluster.remove(maxIdx));
                    subclusterAffinities = getAffinities(subcluster, subcluster);
                    maxIdx = maxIndex(subclusterAffinities);

                    if(!change){
                        change = true;
                    }
                }
            }

            clusterAffinities.add(subclusterAffinities);
            subclusters.add(subcluster);
        }

        return subclusters;
    }

    private double[] getAffinities(ArrayList<Integer> indicies, ArrayList<Integer> subcluster){
        double[] affinities = new double[indicies.size()];

        for (int n = 0; n < affinities.length; n++) {
            for (int i = 0; i < subcluster.size(); i++) {
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

    //Bellaachia, Abdelghani, et al.
    //"E-CAST: a data mining algorithm for gene expression data."
    //Proceedings of the 2nd International Conference on Data Mining in Bioinformatics. Springer-Verlag, 2002.

    private void computeThreshold(ArrayList<Integer> indicies){
        int a = 0;
        int count = 0;

        for (int n = 0; n < indicies.size(); n++) {
            for (int i = 0; i < indicies.size(); i++) {
                double dist;

                if (indicies.get(n) > indicies.get(i)) {
                    dist = distanceMatrix[indicies.get(n)][indicies.get(i)];
                } else {
                    dist = distanceMatrix[indicies.get(i)][indicies.get(n)];
                }

                //may need fixing

                if (dist < 0.5){
                    a += dist - 0.5;
                    count++;
                }
            }
        }

        affinityThreshold = (a/count)+0.5;
    }

    private void normaliseDistanceMatrix(){
        double maxDist = 0;

        for (int i = 0; i < distanceMatrix.length; i++){
            for (int n = 0; n < i; n++){
                if (distanceMatrix[i][n] > maxDist){
                    maxDist = distanceMatrix[i][n];
                }
            }
        }

        for (int i = 0; i < distanceMatrix.length; i++){
            for (int n = 0; n < i; n++){
                distanceMatrix[i][n] = distanceMatrix[i][n]/maxDist;
            }
        }
    }
}
