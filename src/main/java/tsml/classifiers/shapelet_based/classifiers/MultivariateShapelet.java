package tsml.classifiers.shapelet_based.classifiers;

import tsml.classifiers.TSClassifier;
import tsml.classifiers.shapelet_based.filter.ExhaustiveDependantFilter;
import tsml.classifiers.shapelet_based.filter.ExhaustiveIndependantFilter;
import tsml.classifiers.shapelet_based.filter.RandomFilter;
import tsml.classifiers.shapelet_based.transform.ShapeletTransformMV;
import tsml.classifiers.shapelet_based.filter.ShapeletFilterMV;
import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;
import tsml.data_containers.ts_fileIO.TSReader;
import weka.classifiers.Classifier;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

public class MultivariateShapelet implements TSClassifier {

    ShapeletParams params ;

    ShapeletTransformMV transform;

    private TSClassifier classifier;
    private ArrayList<ShapeletMV> shapelets;
    private TimeSeriesInstances transformData;



    public MultivariateShapelet(ShapeletParams params){
        this.params = params;
    }

    @Override
    public Classifier getClassifier() {
        return null;
    }

    @Override
    public void buildClassifier(TimeSeriesInstances data) throws Exception {
        shapelets = this.params.filter.createFilter().findShapelets(params, data,null);
        transformData = transform.transform(data, shapelets);
        classifier.buildClassifier(transformData);

    }


    @Override
    public double[][] distributionForInstances(TimeSeriesInstances data) throws Exception {
        TimeSeriesInstances transformData = transform.transform(data,shapelets);
        return classifier.distributionForInstances(transformData);
    }

    @Override
    public double[] classifyInstances(TimeSeriesInstances data) throws Exception {
        TimeSeriesInstances transformData = transform.transform(data,shapelets);
        return classifier.classifyInstances(transformData);
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

            ShapeletParams params = new ShapeletParams(100,2,23, FilterTypes.EXHAUSTIVE_D);

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

    public enum FilterTypes {
        EXHAUSTIVE_D {
            @Override
            public ShapeletFilterMV createFilter() {
                return new ExhaustiveDependantFilter();
            }
        },
        EXHAUSTIVE_I {
            @Override
            public ShapeletFilterMV createFilter() {
                return new ExhaustiveIndependantFilter();
            }
        },
        RANDOM {
            @Override
            public ShapeletFilterMV createFilter() {
                return new RandomFilter();
            }
        };

        public abstract ShapeletFilterMV createFilter();
    }

    public static class ShapeletParams{
        public int k;
        public int min;
        public int max;
        public FilterTypes filter;

        ShapeletParams(int k, int min, int max, FilterTypes filter){
            this.k = k ;
            this.min = min;
            this.max = max;
            this.filter = filter;
        }

    }
}
