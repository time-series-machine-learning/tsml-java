package tsml.clusterers;

import experiments.data.DatasetLoading;
import tsml.classifiers.dictionary_based.BOSSIndividual;
import tsml.classifiers.dictionary_based.BitWord;
import weka.core.Instances;
import machine_learning.clusterers.PAM;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Map;

import static utilities.ClusteringUtilities.randIndex;
import static utilities.ClusteringUtilities.zNormalise;
import static utilities.InstanceTools.deleteClassAttribute;

/**
 * Class for a dictionary based clustering algorithm, very experimental.
 *
 * @author Matthew Middlehurst
 */
public class DictClusterer extends AbstractTimeSeriesClusterer {

    private int k = 2;
    private int ensembleSize = 50;

    private int seed = Integer.MIN_VALUE;

    private final Integer[] wordLengths = { 16, 14, 12, 10, 8 };
    private final int alphabetSize = 4;
    private final boolean[] norm = { true, false };
    private double maxWinLenProportion = 0.5;

    public DictClusterer(){}

    @Override
    public int numberOfClusters() { return k; }

    public void setNumberOfClusters(int n){ k = n; }

    @Override
    public void buildClusterer(Instances data) throws Exception {
        if (copyInstances){
            data = new Instances(data);
        }

        deleteClassAttribute(data);
        zNormalise(data);

        int[][] parameters = getParameters(data.numAttributes());
        int curWordLen = 0;
        BOSSIndividual boss = null;

        int[][] paramClusters = new int[ensembleSize][];
        double minRandIndex = 1;
        int minIndex = 0;

        //Create a matrix of BOSS distances using varying parameter sets, use the clusters with the best rand index.
        for (int i = 0; i < ensembleSize; i++){
            if (curWordLen == 0) {
                boss = new BOSSIndividual(parameters[i][0], parameters[i][1], parameters[i][2], norm[parameters[i][3]]);
                boss.buildClassifier(data);
            }
            else{
                boss.buildShortenedBags(parameters[i][0]);
            }

            double[][] matrix = matrixFromBags(data, boss);
            PAM pam = new PAM(matrix);
            pam.setNumberOfClusters(k);
            pam.setSeed(seed+i);
            pam.buildClusterer(data);
            paramClusters[i] = pam.getAssignments();

            double randIndex = 1;

            if (i > 0){
                randIndex = 1-randIndex(paramClusters[i-1],paramClusters[i]);
            }

            if (randIndex < minRandIndex){
                minRandIndex = randIndex;
                minIndex = i;
            }


            if (curWordLen == wordLengths.length-1){
                curWordLen = 0;
            }
            else{
                curWordLen++;
            }
        }

        assignments = paramClusters[minIndex];
        System.out.println(Arrays.toString(assignments));
        clusters = new ArrayList[k];

        for (int i = 0; i < k; i++){
            clusters[i] = new ArrayList();
        }

        for (int i = 0; i < data.numInstances(); i++){
            for (int n = 0; n < k; n++){
                if(n == assignments[i]){
                    clusters[n].add(i);
                    break;
                }
            }
        }
    }

    //Selects and returns ensembleSize number of BOSS parameter sets
    private int[][] getParameters(int seriesLength){
        int minWindow = 10;
        int maxWindow = (int)(seriesLength*maxWinLenProportion);
        if (maxWindow < minWindow) minWindow = maxWindow/2;
        int numLengths = (int)Math.ceil(ensembleSize/(wordLengths.length*norm.length));
        int winInc = (maxWindow - minWindow) / numLengths;
        if (winInc < 1) winInc = 1;

        int[][] parameters = new int[ensembleSize][];
        int currentParam = 0;

        for (int normalise = 0; normalise < norm.length && currentParam < ensembleSize; normalise++) {
            for (int winSize = minWindow; winSize <= maxWindow && currentParam < ensembleSize; winSize += winInc) {
                for (Integer wordLen : wordLengths) {
                    int[] paramSet = {wordLen, alphabetSize, winSize, normalise};
                    parameters[currentParam] = paramSet;

                    currentParam++;

                    if (currentParam == ensembleSize){
                        break;
                    }
                }
            }
        }

        return parameters;
    }

    //Turns the bags attribute from BOSS into a 2d double array of BOSS distances
    private double[][] matrixFromBags(Instances data, BOSSIndividual boss){
        double[][] distMatrix = new double[data.numInstances()][];
        ArrayList<BOSSIndividual.Bag> bags = boss.getBags();

        for (int i = 0; i < data.numInstances(); i++){
            distMatrix[i] = new double[i+1];
            BOSSIndividual.Bag first = bags.get(i);

            for (int n = 0; n < i; n++){
                BOSSIndividual.Bag second = bags.get(n);
                double dist = 0;

                ArrayList<Integer> secondSet = new ArrayList();

                for (Map.Entry<BitWord, Integer> entry : first.entrySet()) {
                    Integer valA = entry.getValue();
                    Integer valB = second.get(entry.getKey());

                    if (valB == null) {
                        valB = 0;
                    }
                    else{
                        secondSet.add(valB);
                    }

                    dist += (valA-valB)*(valA-valB);
                }

                for (Map.Entry<BitWord, Integer> entry : second.entrySet()) {
                    if (secondSet.remove(entry.getKey())){
                        continue;
                    }

                    Integer valB = entry.getValue();
                    dist += -valB*-valB;
                }

                distMatrix[i][n] = Math.sqrt(dist);
            }
        }

        return distMatrix;
    }

    public static void main(String[] args) throws Exception{
        String dataset = "Trace";
        Instances inst = DatasetLoading.loadDataNullable("D:\\CMP Machine Learning\\Datasets\\TSC Archive\\" + dataset + "/" + dataset + "_TRAIN.arff");
        Instances inst2 = DatasetLoading.loadDataNullable("D:\\CMP Machine Learning\\Datasets\\TSC Archive\\" + dataset + "/" + dataset + "_TEST.arff");
//        Instances inst = ClassifierTools.loadData("Z:\\Data\\TSCProblems2018\\" + dataset + "/" + dataset + "_TRAIN.arff");
//        Instances inst2 = ClassifierTools.loadData("Z:\\Data\\TSCProblems2018\\" + dataset + "/" + dataset + "_TEST.arff");
        inst.setClassIndex(inst.numAttributes()-1);
        inst.addAll(inst2);

        DictClusterer k = new DictClusterer();
        k.seed = 0;
        k.k = inst.numClasses();
        k.buildClusterer(inst);

        System.out.println(k.clusters.length);
        System.out.println(Arrays.toString(k.clusters));
        System.out.println(randIndex(k.assignments, inst));
    }
}
