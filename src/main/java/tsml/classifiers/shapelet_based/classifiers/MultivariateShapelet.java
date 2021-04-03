package tsml.classifiers.shapelet_based.classifiers;

import machine_learning.classifiers.ensembles.CAWPE;
import machine_learning.classifiers.ensembles.voting.MajorityConfidence;
import machine_learning.classifiers.ensembles.weightings.TrainAcc;
import tsml.classifiers.TSClassifier;
import tsml.classifiers.shapelet_based.distances.ShapeletDistanceEuclidean;
import tsml.classifiers.shapelet_based.distances.ShapeletDistanceFunction;
import tsml.classifiers.shapelet_based.filter.*;
import tsml.classifiers.shapelet_based.quality.GainRatioBinaryQuality;
import tsml.classifiers.shapelet_based.quality.OrderlineQuality;
import tsml.classifiers.shapelet_based.quality.GainRatioQuality;
import tsml.classifiers.shapelet_based.quality.ShapeletQualityFunction;
import tsml.classifiers.shapelet_based.type.ShapeletFunctionsDependant;
import tsml.classifiers.shapelet_based.type.ShapeletFunctionsIndependent;
import tsml.classifiers.shapelet_based.type.ShapeletFunctions;
import tsml.classifiers.shapelet_based.type.ShapeletMV;
import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;
import tsml.data_containers.ts_fileIO.TSReader;
import tsml.transformers.shapelet_tools.Shapelet;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.RotationForest;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.*;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;

public class MultivariateShapelet implements TSClassifier {

    private ShapeletParams params;
    private Classifier classifier;
    private ArrayList<ShapeletMV> shapelets;
    private Instances transformData;

    public static Random RAND = new Random();

    public MultivariateShapelet(ShapeletParams params){
        this.params = params;
    }

    @Override
    public Classifier getClassifier() {
        return null;
    }

    @Override
    public TimeSeriesInstances getTSTrainData() {
        return null;
    }

    @Override
    public void setTSTrainData(TimeSeriesInstances train) {

    }

    @Override
    public void buildClassifier(TimeSeriesInstances data) throws Exception {


        ShapeletFilterMV filter =  this.params.filter.createFilter();
        filter.setHourLimit(4);
        shapelets =filter.findShapelets(params, data);
        System.out.println(shapelets);
        transformData = buildTansformedDataset(data);
        classifier = params.classifier.createClassifier();
        classifier.buildClassifier(transformData);
    }

    @Override
    public double[] distributionForInstance(TimeSeriesInstance data) throws Exception {
        Instance transformData = buildTansformedInstance(data);
        return classifier.distributionForInstance(transformData);
    }

    @Override
    public double classifyInstance(TimeSeriesInstance data) throws Exception {
        Instance transformData = buildTansformedInstance(data);
        return classifier.classifyInstance(transformData);
    }

    private Instances buildTansformedDataset(TimeSeriesInstances instances) {
        Instances output = determineOutputFormat(instances);
        int size = shapelets.size();
        int dataSize = instances.numInstances();

        for (int j = 0; j < dataSize; j++) {
            output.add(new DenseInstance(size + 1));
        }
        ShapeletDistanceFunction distanceFunction = this.params.distance.createShapeletDistance();
        double dist;
        int i=0;
        for (ShapeletMV shapelet: this.shapelets) {
            for (int j = 0; j < dataSize; j++) {
                dist = distanceFunction.calculate(shapelet,instances.get(j));
                output.instance(j).setValue(i, dist);
            }
            i++;
        }

        for (int j = 0; j < dataSize; j++) {
            output.instance(j).setValue(size, instances.get(j).getTargetValue());
        }

        return output;
    }

    private Instance buildTansformedInstance(TimeSeriesInstance instance) {
        Shapelet s;
        int size = shapelets.size();
        Instance out = new DenseInstance(size + 1);


        out.setDataset(transformData);
        double dist;
        int i=0;
        for (ShapeletMV shapelet: this.shapelets) {
            dist = this.params.distance.createShapeletDistance().calculate(shapelet,instance);
            out.setValue(i, dist);
            i++;
        }
        return out;
    }

    private Instances determineOutputFormat(TimeSeriesInstances inputFormat) throws IllegalArgumentException {

        if (this.shapelets.size() < 1) {
            throw new IllegalArgumentException("ShapeletTransform not initialised correctly - " +
                    "please specify a value of k (this.numShapelets) that is greater than or equal to 1. " +
                    "It is currently set tp "+this.shapelets.size());
        }

        int length = this.shapelets.size();
        ArrayList<Attribute> atts = new ArrayList<>();
        String name;
        for (int i = 0; i < length; i++) {
            name = "Shapelet_" + i;
            atts.add(new Attribute(name));
        }

        FastVector vals = new FastVector(inputFormat.numClasses());
        for (int i = 0; i < inputFormat.numClasses(); i++) {
            vals.addElement(inputFormat.getClassLabels()[i]);
        }
        atts.add(new Attribute("Target", vals));

        Instances result = new Instances("Shapelets" + inputFormat.getProblemName(), atts, inputFormat.numInstances());
        result.setClassIndex(result.numAttributes() - 1);
        return result;
    }



    public enum ShapeletFilters {
        EXHAUSTIVE {
            @Override
            public ShapeletFilterMV createFilter() {
                return new ExhaustiveFilter();
            }
        },
        RANDOM {
            @Override
            public ShapeletFilterMV createFilter() {
                return new RandomFilter();
            }
        },
        RANDOM_BY_CLASS {
            @Override
            public ShapeletFilterMV createFilter() {
                return new RandomFilterByClass();
            }
        };;

        public abstract ShapeletFilterMV createFilter();
    }

    public enum ShapeletQualities {
        GAIN_RATIO {
            @Override
            public ShapeletQualityFunction createShapeletQuality(TimeSeriesInstances instances,
                                                                 ShapeletDistanceFunction distance) {
                return new GainRatioQuality(instances,distance);
            }
        },
        ORDER_LINE_AARON {
            @Override
            public ShapeletQualityFunction createShapeletQuality(TimeSeriesInstances instances,
                                                                 ShapeletDistanceFunction distance) {
                return new OrderlineQuality(instances,distance);
            }
        },
        BINARY{
            public ShapeletQualityFunction createShapeletQuality(TimeSeriesInstances instances,
                                                                 ShapeletDistanceFunction distance) {
                return new GainRatioBinaryQuality(instances,distance);
            }
        };

        public abstract ShapeletQualityFunction createShapeletQuality(TimeSeriesInstances instances,
                                                             ShapeletDistanceFunction distance) ;
    }

    public enum ShapeletDistances {
        EUCLIDEAN {
            @Override
            public ShapeletDistanceFunction createShapeletDistance() {
                return new ShapeletDistanceEuclidean();
            }
        },
        NEW {
            @Override
            public ShapeletDistanceFunction createShapeletDistance() {
                return new ShapeletDistanceEuclidean();
            }
        };

        public abstract ShapeletDistanceFunction createShapeletDistance();
    }

    public enum ShapeletFactories {
        DEPENDANT {
            @Override
            public ShapeletFunctions createShapeletType() {
                return new ShapeletFunctionsDependant();
            }
        },
        INDEPENDENT {
            @Override
            public ShapeletFunctions createShapeletType() {
                return new ShapeletFunctionsIndependent();
            }
        };

        public abstract ShapeletFunctions createShapeletType();
    }

    public enum AuxClassifiers {
        ENSEMBLE {
            @Override
            public Classifier createClassifier() {
                CAWPE ensemble=new CAWPE();
                ensemble.setWeightingScheme(new TrainAcc(4));
                ensemble.setVotingScheme(new MajorityConfidence());
                Classifier[] classifiers = new Classifier[7];
                String[] classifierNames = new String[7];

                SMO smo = new SMO();
                smo.turnChecksOff();
                smo.setBuildLogisticModels(true);
                PolyKernel kl = new PolyKernel();
                kl.setExponent(2);
                smo.setKernel(kl);
                classifiers[0] = smo;
                classifierNames[0] = "SVMQ";

                RandomForest r=new RandomForest();
                r.setNumTrees(500);
                classifiers[1] = r;
                classifierNames[1] = "RandF";


                RotationForest rf=new RotationForest();
                rf.setNumIterations(100);
                classifiers[2] = rf;
                classifierNames[2] = "RotF";
                IBk nn=new IBk();
                classifiers[3] = nn;
                classifierNames[3] = "NN";
                NaiveBayes nb=new NaiveBayes();
                classifiers[4] = nb;
                classifierNames[4] = "NB";
                J48 c45=new J48();
                classifiers[5] = c45;
                classifierNames[5] = "C45";
                SMO svml = new SMO();
                svml.turnChecksOff();
                svml.setBuildLogisticModels(true);
                PolyKernel k2 = new PolyKernel();
                k2.setExponent(1);
                smo.setKernel(k2);
                classifiers[6] = svml;
                classifierNames[6] = "SVML";
                ensemble.setClassifiers(classifiers, classifierNames, null);

                return  ensemble;
            }
        },
        LINEAR {
            @Override
            public Classifier createClassifier() {
                return new LibLINEARN();

            }
        },
        SVM {
            @Override
            public Classifier createClassifier() {
                SMO svml = new SMO();
                svml.turnChecksOff();
                svml.setBuildLogisticModels(true);
                PolyKernel k2 = new PolyKernel();
                k2.setExponent(1);
                svml.setKernel(k2);
                return  svml;

            }
        },
        ROT{
            @Override
            public Classifier createClassifier() {
                RotationForest rf = new RotationForest();
                rf.setNumIterations(100);

                return rf;
            }
        };

        public abstract Classifier createClassifier();
    }

    public static class ShapeletParams{
        public int k;
        public int min;
        public int max;
        public int maxIterations;
        public double minDist;
        public ShapeletFilters filter;
        public ShapeletQualities quality;
        public ShapeletDistances distance;
        public ShapeletFactories type;
        public AuxClassifiers classifier;

        public ShapeletParams(int k, int min, int max, int maxIterations, double minDist,
                              ShapeletFilters filter, ShapeletQualities quality,
                              ShapeletDistances distance, ShapeletFactories type,
                              AuxClassifiers classifier){
            this.k = k ;
            this.min = min;
            this.max = max;
            this.maxIterations = maxIterations;
            this.minDist = minDist;
            this.filter = filter;
            this.quality = quality;
            this.distance = distance;
            this.type = type;
            this.classifier = classifier;
        }

    }

    public static void main(String[] arg){
        String m_local_path = "C:\\Users\\fbu19zru\\code\\Multivariate_ts\\";

        String dataset = "BasicMotions";
        String filepath = m_local_path + dataset + "\\" + dataset;

        TSReader ts_reader = null;
        try {
            ts_reader = new TSReader(new FileReader(new File(filepath + "_TRAIN" + ".ts")));
            TimeSeriesInstances ts_train_data = ts_reader.GetInstances();

            ts_reader = new TSReader(new FileReader(new File(filepath + "_TEST" + ".ts")));
            TimeSeriesInstances ts_test_data = ts_reader.GetInstances();
            int f = ts_train_data.numInstances()*ts_train_data.getMaxLength()*ts_train_data.getMaxNumChannels();
            System.out.println( "f= " + f);
            ShapeletParams params = new ShapeletParams(f,3,ts_train_data.getMaxLength()/2,
                    10000,0.01,
                    ShapeletFilters.RANDOM, ShapeletQualities.GAIN_RATIO, ShapeletDistances.EUCLIDEAN,
                    ShapeletFactories.DEPENDANT,
                    AuxClassifiers.ENSEMBLE);

            MultivariateShapelet shapelet = new MultivariateShapelet(params);
            shapelet.buildClassifier(ts_train_data);

            double ok=0, wrong=0;
            for (TimeSeriesInstance ts: ts_test_data){
                double pred = shapelet.classifyInstance(ts);
                if (ts.getTargetValue()==pred){
                    ok++;
                }else{
                    wrong++;
                }
            }
            System.out.println("Acc= " + ok/(ok+wrong));


        } catch (IOException e) {
            e.printStackTrace();
        } catch (Exception e) {
            e.printStackTrace();
        }


    }
}
