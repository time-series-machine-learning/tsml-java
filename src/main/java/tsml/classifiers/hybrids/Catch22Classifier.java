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
package tsml.classifiers.hybrids;

import experiments.data.DatasetLoading;
import tsml.classifiers.EnhancedAbstractClassifier;
import tsml.transformers.Catch22;
import utilities.ClassifierTools;
import weka.classifiers.Classifier;
import weka.classifiers.trees.RandomForest;
import weka.core.*;

import java.util.ArrayList;
import java.util.concurrent.TimeUnit;

import static utilities.InstanceTools.resampleTrainAndTestInstances;
import static utilities.multivariate_tools.MultivariateInstanceTools.*;

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
    private int numColumns;

    public Catch22Classifier(){
        super(CANNOT_ESTIMATE_OWN_PERFORMANCE);
        ((RandomForest)cls).setNumTrees(500);
    }

    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        result.setMinimumNumberInstances(2);

        // attributes
        result.enable(Capabilities.Capability.RELATIONAL_ATTRIBUTES);
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);

        // class
        result.enable(Capabilities.Capability.NOMINAL_CLASS);

        return result;
    }

    public void setClassifier(Classifier cls){
        this.cls = cls;
    }

    public void setNormalise(boolean b) { this.norm = b; }

    public void setOutlierNormalise(boolean b) { this.outlierNorm = b; }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        super.buildClassifier(data);
        trainResults.setBuildTime(System.nanoTime());
        getCapabilities().testWithFail(data);

        Instances[] columns;
        //Multivariate
        if (data.checkForAttributeType(Attribute.RELATIONAL)) {
            columns = splitMultivariateInstances(data);
            numColumns = numDimensions(data);
        }
        //Univariate
        else{
            columns = new Instances[]{data};
            numColumns = 1;
        }

        c22 = new Catch22();
        c22.setNormalise(norm);
        c22.setOutlierNormalise(outlierNorm);

        ArrayList<Attribute> atts = new ArrayList<>();
        for (int i = 1; i <= 22*numColumns; i++){
            atts.add(new Attribute("att" + i));
        }
        atts.add(data.classAttribute());
        Instances transformedData = new Instances("Catch22Transform", atts, data.numInstances());
        transformedData.setClassIndex(transformedData.numAttributes()-1);
        header = new Instances(transformedData,0);

        //transform each dimension using the catch22 transformer into a sincle vector
        for (int i = 0 ; i < data.numInstances(); i++){
            double[] d = new double[transformedData.numAttributes()];
            for (int n = 0 ; n < numColumns; n++){
                Instance inst = (c22.transform(columns[n].get(i)));
                for (int j = 0; j < 22; j++){
                    d[n * 22 + j] = inst.value(j);
                }
            }
            d[transformedData.numAttributes()-1] = data.get(i).classValue();
            transformedData.add(new DenseInstance(1, d));
        }

        if (cls instanceof Randomizable){
            ((Randomizable) cls).setSeed(seed);
        }

        cls.buildClassifier(transformedData);

        trainResults.setTimeUnit(TimeUnit.NANOSECONDS);
        trainResults.setBuildTime(System.nanoTime() - trainResults.getBuildTime());
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        return cls.classifyInstance(predictionTransform(instance));
    }

    public double[] distributionForInstance(Instance instance) throws Exception {
        return cls.distributionForInstance(predictionTransform(instance));
    }

    public Instance predictionTransform(Instance instance){
        Instance[] columns;
        //Multivariate
        if (numColumns > 1) {
            columns = splitMultivariateInstance(instance);
        }
        //Univariate
        else{
            columns = new Instance[]{instance};
        }

        //transform each dimension using the catch22 transformer into a sincle vector
        double[] d = new double[header.numAttributes()];
        for (int n = 0 ; n < numColumns; n++){
            Instance inst = (c22.transform(columns[n]));
            for (int j = 0; j < 22; j++){
                d[n * 22 + j] = inst.value(j);
            }
        }
        d[header.numAttributes()-1] = instance.classValue();
        Instance transformedInst = new DenseInstance(1, d);
        transformedInst.setDataset(header);

        return transformedInst;
    }

    public static void main(String[] args) throws Exception {
        int fold = 0;
        String dataset = "ItalyPowerDemand";

        Instances train = DatasetLoading.loadDataNullable("Z:\\ArchiveData\\Univariate_arff\\" + dataset
                + "\\" + dataset + "_TRAIN.arff");
        Instances test = DatasetLoading.loadDataNullable("Z:\\ArchiveData\\Univariate_arff\\" + dataset
                + "\\" + dataset + "_TEST.arff");
        Instances[] data = resampleTrainAndTestInstances(train, test, fold);
        train = data[0];
        test = data[1];

        Catch22Classifier c;
        double accuracy;

        c = new Catch22Classifier();
        c.buildClassifier(train);
        accuracy = ClassifierTools.accuracy(test, c);

        System.out.println("Catch22Classifier accuracy on " + dataset + " fold " + fold + " = " + accuracy);
    }
}
