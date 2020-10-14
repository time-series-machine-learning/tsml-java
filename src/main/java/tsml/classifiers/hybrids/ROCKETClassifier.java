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
import tsml.transformers.ROCKET;
import utilities.ClassifierTools;
import weka.classifiers.Classifier;
import weka.classifiers.functions.Logistic;
import weka.classifiers.trees.J48;
import weka.core.*;

import java.util.ArrayList;

import static utilities.InstanceTools.resampleTrainAndTestInstances;
import static utilities.multivariate_tools.MultivariateInstanceTools.*;

/**
 * @author Matthew Middlehurst
 */
public class ROCKETClassifier extends EnhancedAbstractClassifier {

    private Classifier cls = new Logistic();
    private ROCKET rocket;
    private Instances header;

    public ROCKETClassifier(){
        super(CANNOT_ESTIMATE_OWN_PERFORMANCE);
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

    @Override
    public void buildClassifier(Instances data) throws Exception {
        super.buildClassifier(data);
        trainResults.setBuildTime(System.nanoTime());
        getCapabilities().testWithFail(data);

        rocket = new ROCKET();
        rocket.setSeed(seed);

        Instances transformedData = rocket.fitTransform(data);
        header = new Instances(transformedData,0);

        if (cls instanceof Randomizable){
            ((Randomizable) cls).setSeed(seed);
        }

        cls.buildClassifier(transformedData);
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
        Instance transformedInst = rocket.transform(instance);
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

        ROCKETClassifier c;
        double accuracy;

        c = new ROCKETClassifier();
        c.seed = fold;
        c.buildClassifier(train);
        accuracy = ClassifierTools.accuracy(test, c);

        System.out.println("ROCKETClassifier accuracy on " + dataset + " fold " + fold + " = " + accuracy);
    }
}
