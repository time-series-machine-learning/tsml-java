package ml_6002b_coursework;

import weka.classifiers.AbstractClassifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Random;


public class TreeEnsemble extends AbstractClassifier{

    private int numTrees;

    private CourseworkTree[] trees;
    private double attributeProportion;
//    private HashMap<ArrayList><CourseworkTree> treeAttributes = new HashMap;
    private boolean averageDistribution = false;

    public TreeEnsemble() {
        //default values
        trees = new CourseworkTree[numTrees];
        //treeAttributes= new ArrayList[numTrees];
        this.numTrees = 50;
        this.attributeProportion = 0.5;
    }

    public void setNumTrees(int numTrees) {
        this.numTrees = numTrees;
    }

    public void setAttributeProportion(double attributeProportion) {
        this.attributeProportion = attributeProportion;
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        Random splitRnd = new Random();
        Random rnd = new Random();
        trees = new CourseworkTree[numTrees];
        for (int i=0;i<numTrees;i++){
            ArrayList<Integer> attributes = new ArrayList<>();
            trees[i] = new CourseworkTree();
            trees[i].buildClassifier(data);

            String[] splitMeasure = new String[]{"infoGain", "infoGainRatio", "gini", "chiSquared"};
            trees[i].setOptions(splitMeasure[splitRnd.nextInt(4)]);
            for (int a=0;a<data.numAttributes();a++){
                attributes.add(a);
            }
//            while(attributes.size() > data.numAttributes()*attributeProportion){
//                int randomInt = rnd.nextInt(attributes.size());
//                attributes.remove(randomInt);
//            }
//            System.out.println(attributes);

            int[] arr = attributes.stream().mapToInt(y -> y).toArray();
            //System.out.println(Arrays.toString(arr));

            Remove removeFilter = new Remove();
            removeFilter.setAttributeIndicesArray(arr);
            removeFilter.setInvertSelection(true);
            removeFilter.setInputFormat(data);
            Instances newData = Filter.useFilter(data, removeFilter);

            data.setClassIndex(data.numAttributes() - 1);
            newData.setClassIndex(newData.numAttributes() - 1);


            trees[i].buildClassifier(newData);
            //treeAttributes[i] = attributes;
//            System.out.println(trees[i]);
//            treeAttributes.put(attributes, trees[i]);
        }
    }

    @Override
    public double classifyInstance(Instance instance) {
        double bestClass = 0;

        if (!averageDistribution) {
            int[] count = new int[instance.numClasses()];
            for (int i = 0; i < numTrees; i++) {
                double predicted = trees[i].classifyInstance(instance);
                count[(int) predicted]++;
            }
            for (int i = 0; i < count.length-1; i++) {
                if (bestClass < count[i]) {
                    bestClass = i;
                }
            }

        } else {
            double[] probabilities = distributionForInstance(instance);
            for(int i=0;i<probabilities.length;i++){
                if(probabilities[i] > bestClass){
                    bestClass = i;
                }
            }
        }
//        System.out.println(bestClass+ "bestclass");
        return bestClass;
    }

    @Override
    public double[] distributionForInstance(Instance instance){
        double[] distribution = new double[instance.numClasses()];
        for (int i=0;i<numTrees;i++){

            int prediction = (int) trees[i].classifyInstance(instance);    // <-- every time this is called
//            System.out.println(prediction);
//            System.out.println(trees[1].classifyInstance(instance));
//            System.out.println(trees[0].classifyInstance(instance));
//            System.out.println(trees[i]);
            distribution[prediction]++;
//            System.out.println(Arrays.toString(distribution));
        }

        //convert count to proportion
        for(int i=0; i<distribution.length;i++){
//            System.out.println(numTrees+"numtrees");
            distribution[i] = distribution[i] / (double) numTrees;
//            System.out.println(distribution.length+"len");
//            System.out.println(Arrays.toString(distribution));
        }

        return distribution;
    }

//    @Override
//    public double[] distributionForInstance(Instance inst) throws Exception {
//        double[] probs = new double[inst.numClasses()];
//        for (Classifier trees[i]:){
//            double[] d = c.distributionForInstance(inst);
//            for(int i=0;i<d.length;i++)
//                probs[i]+=d[i];
//        }
//        double sum=0;
//        for(int i=0;i<probs.length;i++)
//            sum+=probs[i];
//        for(int i=0;i<probs.length;i++)
//            probs[i]/=sum;
//        return probs;
//
//    }

    @Override
    public void setOptions(String[] options) throws Exception {
        super.setOptions(options);
        switch (Arrays.toString(options)){
            case "true":
                averageDistribution = true;
                break;
            case "false":
                averageDistribution = false;
                break;
        }
    }


    public static void main(String[] args) throws Exception {
        String[] files = {
                "./src/main/java/ml_6002b_coursework/test_data/optdigits.arff",
                "./src/main/java/ml_6002b_coursework/test_data/Chinatown.arff"
        };


        for (String file : files) {
            FileReader reader = new FileReader(file);
            Instances data = new Instances(reader);
            data.setClassIndex(data.numAttributes() - 1);


            data.randomize(new java.util.Random(0));
            Instances trainData = new Instances(data, 0, (int) (data.numInstances() * 0.75));
            Instances testData = new Instances(data, (int) (data.numInstances() * 0.75), (int) (data.numInstances() * 0.25));
            TreeEnsemble treeEnsemble = new TreeEnsemble();
            treeEnsemble.buildClassifier(trainData);
            int correct = 0;
            int total = 0;
            for(int i=0;i<testData.numInstances();i++){
                Instance instance = testData.instance(i);
                double prediction = treeEnsemble.classifyInstance(instance);

                if(i < 5){
                    double[] dist = treeEnsemble.distributionForInstance(instance);
                    int j = i;
                    j++;
                    System.out.println("Tree Ensemble "+j+" with "+data.relationName()+" has the probability = "
                            +Arrays.toString(dist));
                }
                if(instance.classValue() == prediction){
                    correct++;
                }
                total++;
            }
            double accuracy = (double) correct / total;
            System.out.println(data.relationName()+" has test accuracy = "+accuracy*100);
        }
    }
}





