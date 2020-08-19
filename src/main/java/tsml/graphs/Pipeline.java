package tsml.graphs;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import tsml.classifiers.EnhancedAbstractClassifier;
import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;
import tsml.data_containers.utilities.Splitter;
import tsml.data_containers.utilities.TimeSeriesSummaryStatistics;
import tsml.transformers.Transformer;
import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

public class Pipeline extends EnhancedAbstractClassifier {

    List<Layer> layers;

    public Pipeline() {
        layers = new ArrayList<>();
    }

    public void add(String name, Transformer transformer) {
        layers.add(new TransformerLayer(name, transformer));
    }

    public void add(String name, EnhancedAbstractClassifier clf) {
        layers.add(new ClassifierLayer<EnhancedAbstractClassifier>(name, clf));
    }

    public void add(String name, AbstractClassifier clf) {
        layers.add(new ClassifierLayer<EnhancedAbstractClassifier>(name, new EnhancedClassifierWrapper(clf)));
    }

    public void concat(String name, Layer... concats){
        layers.add(new ConcatLayer(name, concats));
    }

    public void split(String name, Pipeline... models) {
        layers.add(new PipelineLayer(name, models));
        layers.add(new EnsembleLayer());
	}

    @Override
    public void buildClassifier(TimeSeriesInstances trainData) throws Exception {
        //super.buildClassifier(trainData);

        TimeSeriesInstances data = trainData;
        for (Layer layer : layers) {
            data = layer.fit(data);
        }   
    }

    public TimeSeriesInstances fit(TimeSeriesInstances trainData) throws Exception {
        TimeSeriesInstances data = trainData;
        for (Layer layer : layers) {
            data = layer.fit(data);
        }  
        return data;
    }

    public TimeSeriesInstances predict(TimeSeriesInstances testData)throws Exception{
        TimeSeriesInstances data = testData;
        for (Layer layer : layers) {
            data = layer.predict(data);
        }   
        return data;
    }

    @Override
    public double[][] distributionForInstances(TimeSeriesInstances testData) throws Exception {
        return predict(testData).getHSliceArray(0);
    }

    public static abstract class Layer {
        String name;

        abstract TimeSeriesInstances fit(TimeSeriesInstances input) throws Exception;
        abstract TimeSeriesInstances predict(TimeSeriesInstances inst) throws Exception;
    }

    public static class ClassifierLayer<T extends EnhancedAbstractClassifier> extends Layer {
        EnhancedAbstractClassifier classifier;
        boolean fit;

        public ClassifierLayer(String name, T clf) {
            this.name = name;
            this.classifier = clf;
        }

        TimeSeriesInstances fit(TimeSeriesInstances input) throws Exception {
            classifier.buildClassifier(input);
            return predict(input);
        }

        @Override
        TimeSeriesInstances predict(TimeSeriesInstances data) throws Exception {            
            return new TimeSeriesInstances(new double[][][]{classifier.distributionForInstances(data)}, data.getClassIndexes(), data.getClassLabels());
        }
    }

    public static class PipelineLayer extends Layer {
        Pipeline[] pipelines;

        public PipelineLayer(String name, Pipeline... pipelines) {
            this.name = name;
            this.pipelines = pipelines;
        }

        @Override
        TimeSeriesInstances fit(TimeSeriesInstances input) throws Exception{
            List<TimeSeriesInstances> split = Splitter.splitTimeSeriesInstances(input);

            if (pipelines.length != split.size()) {
                System.out.println("layers Split MisMatch");
            }
            List<TimeSeriesInstances> t_split = new ArrayList<TimeSeriesInstances>(split.size());

            for (int i = 0; i < pipelines.length; i++) {
                t_split.add(pipelines[i].fit(split.get(i)));
            }

            return Splitter.mergeTimeSeriesInstances(t_split);
        }

        @Override
        TimeSeriesInstances predict(TimeSeriesInstances inst) throws Exception {
            System.out.println(inst);

            List<TimeSeriesInstances> split = Splitter.splitTimeSeriesInstances(inst);

            if (pipelines.length != split.size()) {
                System.out.println("layers Split MisMatch");
            }
            List<TimeSeriesInstances> t_split = new ArrayList<TimeSeriesInstances>(split.size());

            for (int i = 0; i < pipelines.length; i++) {
                t_split.add(pipelines[i].predict(split.get(i)));
            }

            return Splitter.mergeTimeSeriesInstances(t_split);
        }
    }

    public static class TransformerLayer extends Layer {
        Transformer transformer;

        public TransformerLayer(String name, Transformer clf) {
            this.name = name;
            this.transformer = clf;
        }

        @Override
        TimeSeriesInstances fit(TimeSeriesInstances input) throws Exception{
            return this.transformer.transform(input);
        }

        @Override
        TimeSeriesInstances predict(TimeSeriesInstances data) throws Exception {
            return this.transformer.transform(data);
        }
    }

    public static class ConcatLayer extends Layer {
        Layer[] layers;

        public ConcatLayer(String name, Layer... layers) {
            this.name = name;
            this.layers = layers;
        }

        @Override
        TimeSeriesInstances fit(TimeSeriesInstances input) throws Exception{
            return predict(input);
        }

        @Override
        TimeSeriesInstances predict(TimeSeriesInstances inst) throws Exception {
            List<TimeSeriesInstances> split = Splitter.splitTimeSeriesInstances(inst);

            if (layers.length != split.size()) {
                System.out.println("layers Split MisMatch");
            }
            List<TimeSeriesInstances> t_split = new ArrayList<TimeSeriesInstances>(split.size());

            for (int i = 0; i < layers.length; i++) {
                t_split.add(layers[i].predict(split.get(i)));
            }

            return Splitter.mergeTimeSeriesInstances(t_split);
        }
    }

    public static class EnsembleLayer extends Layer {
        @Override
        TimeSeriesInstances fit(TimeSeriesInstances input) {
            return predict(input);
        }

        @Override
        TimeSeriesInstances predict(TimeSeriesInstances data) {
            double[][][] output = new double[data.numInstances()][][];
            for(int j=0; j< data.numInstances(); j++){
                double[][] output1 = new double[1][data.get(j).getMaxLength()];
                for(int i=0; i<data.get(j).getMaxLength(); i++){
                    output1[0][i] = TimeSeriesSummaryStatistics.mean(data.get(j).getSingleVSliceArray(i));
                }

                output[j] = output1;
            }
            return new TimeSeriesInstances(output, data.getClassIndexes(), data.getClassLabels());
        }
    }

    public static class EnhancedClassifierWrapper extends EnhancedAbstractClassifier{

        AbstractClassifier classifier;

        public EnhancedClassifierWrapper(AbstractClassifier clf){
            classifier = clf;
        }

        @Override
        public void buildClassifier(Instances trainData) throws Exception {
            super.buildClassifier(trainData);
            classifier.buildClassifier(trainData);
        }

        @Override
        public double[] distributionForInstance(Instance instance) throws Exception {
            return classifier.distributionForInstance(instance);
        }
    }


}