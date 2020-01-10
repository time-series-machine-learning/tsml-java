package tsml.classifiers.hybrids;

import experiments.data.DatasetLists;
import experiments.data.DatasetLoading;
import tsml.classifiers.EnhancedAbstractClassifier;
import tsml.transformers.Catch22;
import utilities.ClassifierTools;
import weka.classifiers.Classifier;
import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Randomizable;

import static utilities.InstanceTools.resampleTrainAndTestInstances;

/**
 * Classifier built using catch22 features.
 *
 * C.H. Lubba, S.S. Sethi, P. Knaute, S.R. Schultz, B.D. Fulcher, N.S. Jones.
 * catch22: CAnonical Time-series CHaracteristics.
 * Data Mining and Knowledge Discovery (2019)
 *
 * Implementation based on C and Matlab code provided on authors github:
 * https://github.com/chlubba/catch22
 *
 * @author Matthew Middlehurst
 */
public class Catch22Classifier extends EnhancedAbstractClassifier {

    //z-norm before transform
    private boolean norm = false;
    //specifically normalise for the outlier stats, which can take a long time with large positive/negative values
    private boolean outlierNorm = false;

    private Classifier cls = new RandomForest();
    private Catch22 c22;
    private Instances header;

    public Catch22Classifier(){
        super(CANNOT_ESTIMATE_OWN_PERFORMANCE);
    }

    public void setClassifier(Classifier cls){
        this.cls = cls;
    }

    public void setNormalise(boolean b) { this.norm = b; }

    public void setOutlierNormalise(boolean b) { this.outlierNorm = b; }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        trainResults.setBuildTime(System.nanoTime());
        c22 = new Catch22();
        c22.setNormalise(norm);
        c22.setOutlierNormalise(outlierNorm);

        if (cls instanceof Randomizable){
            ((Randomizable) cls).setSeed(seed);
        }

        Instances transformedData = c22.determineOutputFormat(data);
        header = new Instances(transformedData,0);

        for (Instance inst : data){
            transformedData.add(c22.transform(inst));
        }

        cls.buildClassifier(transformedData);
        trainResults.setBuildTime(System.nanoTime() - trainResults.getBuildTime());
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        Instance transformedInst = c22.transform(instance);
        transformedInst.setDataset(header);
        return cls.classifyInstance(transformedInst);
    }

    public double[] distributionForInstance(Instance instance) throws Exception {
        Instance transformedInst = c22.transform(instance);
        transformedInst.setDataset(header);
        return cls.distributionForInstance(transformedInst);
    }

    public static void main(String[] args) throws Exception {
        int fold = 0;
        for (int i = 0; i < DatasetLists.tscProblems112.length; i++) {
            String dataset = DatasetLists.tscProblems112[i];
            Instances train = DatasetLoading.loadDataNullable("Z:\\ArchiveData\\Univariate_arff\\" + dataset + "\\" + dataset + "_TRAIN.arff");
            Instances test = DatasetLoading.loadDataNullable("Z:\\ArchiveData\\Univariate_arff\\" + dataset + "\\" + dataset + "_TEST.arff");
            Instances[] data = resampleTrainAndTestInstances(train, test, fold);
            train = data[0];
            test = data[1];

            Catch22Classifier c;
            double accuracy;

            c = new Catch22Classifier();
            RandomForest rf = new RandomForest();
            rf.setNumTrees(100);
            rf.setSeed(0);
            c.setClassifier(rf);
            c.buildClassifier(train);
            accuracy = ClassifierTools.accuracy(test, c);

            System.out.println("Catch22 accuracy on " + i + " " + dataset + " fold " + fold + " = " + accuracy);
        }
    }
}
