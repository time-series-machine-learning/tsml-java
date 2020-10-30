package tsml.graphs;

import java.util.ArrayList;
import java.util.List;

import tsml.classifiers.EnhancedAbstractClassifier;
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

    public void concat(String name, int[][] inds, Layer... concats){
        layers.add(new ConcatLayer(name, concats, inds));
    }

    public void concat(String name, Pipeline... models){
        layers.add(new ConcatLayer(name, models));
    }

    public void concat(String name, int[][] inds, Pipeline... concats){
        layers.add(new ConcatLayer(name, concats, inds));
    }

    public void concat(String name, Transformer... concats){
        layers.add(new ConcatLayer(name, concats));
    }

    public void concat(String name, int[][] inds, Transformer... concats){
        layers.add(new ConcatLayer(name, concats, inds));
    }
    
    public void split(String name, Layer... splits){
        layers.add(new SplitLayer(name, splits));
    }

    public void split(String name,int[][] slicingIndexes, Layer... splits){
        layers.add(new SplitLayer(name, splits, slicingIndexes));
    }

    public void split(String name, Pipeline... models){
        layers.add(new SplitLayer(name, models));
    }

    public void split(String name,int[][] slicingIndexes, Pipeline... models){
        layers.add(new SplitLayer(name, models, slicingIndexes));
    }

    public void split(String name,int[][] slicingIndexes, Transformer... transforms){
        layers.add(new SplitLayer(name, transforms, slicingIndexes));
    }

    public void split(String name,Transformer... transforms){
        layers.add(new SplitLayer(name, transforms));
    }

    public void splitAndEnsemble(String name, Pipeline... models){
        layers.add(new SplitLayer(name, models));
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
        Pipeline pipeline;

        public PipelineLayer(String name, Pipeline pipeline) {
            this.name = name;
            this.pipeline = pipeline;
        }

        @Override
        TimeSeriesInstances fit(TimeSeriesInstances input) throws Exception{
            return this.pipeline.fit(input);
        }

        @Override
        TimeSeriesInstances predict(TimeSeriesInstances inst) throws Exception {
            return this.pipeline.predict(inst);
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

    public abstract static class MultiLayer extends Layer {
        Layer[] layers;
        int[][] slicingIndexes;

        public MultiLayer(String name, Layer... layers) {
            this.name = name;
            this.layers = layers;
            generate_indexes();
        }

        public MultiLayer(String name, Layer[] layers, int[][] indexes) {
            this.name = name;
            this.layers = layers;
            this.slicingIndexes = indexes;
        }

        public MultiLayer(String name, Transformer[] concats) {
            this.layers = new TransformerLayer[concats.length];
            this.name = name;

            int i=0;
            for(Transformer t : concats)
                layers[i++] = new TransformerLayer(name + "_" + (i-1),t);

            generate_indexes();
        }

        public MultiLayer(String name, Transformer[] concats, int[][] indexes) {
            this.slicingIndexes = indexes;
            this.layers = new TransformerLayer[concats.length];

            int i=0;
            for(Transformer t : concats)
                layers[i++] = new TransformerLayer(name + "_" + (i-1),t);
        }
        public MultiLayer(String name, Pipeline[] models) {
            this.layers = new PipelineLayer[models.length];
            this.name = name;

            int i=0;
            for(Pipeline t : models)
                layers[i++] = new PipelineLayer(name + "_" + (i-1),t);

            generate_indexes();
        }

        public MultiLayer(String name, Pipeline[] models, int[][] indexes) {
            this.slicingIndexes = indexes;
            this.layers = new PipelineLayer[models.length];
            this.name = name;

            int i=0;
            for(Pipeline t : models)
                layers[i++] = new PipelineLayer(name + "_" + (i-1),t);
        }
        
        private void generate_indexes() {           
            slicingIndexes = new int[layers.length][1];
            for(int i=0; i< slicingIndexes.length; i++)
                slicingIndexes[i] = new int[]{i};
        }
    }

    public static class ConcatLayer extends MultiLayer{

        public ConcatLayer(String name, Layer[] layers) {
            super(name, layers);
        }

        public ConcatLayer(String name, Pipeline[] models) {
            super(name, models);
        }

        public ConcatLayer(String name, Layer[] layers, int[][] indexes){
            super(name, layers, indexes);
        }

        public ConcatLayer(String name, Transformer[] concats) {
            super(name, concats);
        }

        public ConcatLayer(String name, Pipeline[] models, int[][] indexes) {
            super(name, models, indexes);
        }

        public ConcatLayer(String name, Transformer[] transforms, int[][] inds) {
            super(name, transforms, inds);
        }

        @Override
        TimeSeriesInstances fit(TimeSeriesInstances inst) throws Exception{
            List<TimeSeriesInstances> split = Splitter.splitTimeSeriesInstances(inst);

            if (layers.length != split.size()) {
                System.out.println("layers Split MisMatch");
            }
            List<TimeSeriesInstances> t_split = new ArrayList<TimeSeriesInstances>(split.size());

            for (int i = 0; i < layers.length; i++) {
                t_split.add(layers[i].fit(split.get(i)));
            }

            return Splitter.mergeTimeSeriesInstances(t_split);
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

    public static class SplitLayer extends MultiLayer {
        
        public SplitLayer(String name, Layer[] layers) {
            super(name, layers);
        }

        public SplitLayer(String name, Layer[] splits, int[][] slicingIndexes) {
            super(name, splits, slicingIndexes);
        }

        public SplitLayer(String name, Pipeline[] models) {
            super(name, models);
        }

        public SplitLayer(String name, Pipeline[] models, int[][] slicingIndexes) {
            super(name, models, slicingIndexes);
        }

        public SplitLayer(String name, Transformer[] transforms, int[][] slicingIndexes) {
            super(name, transforms, slicingIndexes);
        }

        public SplitLayer(String name, Transformer[] transforms) {
            super(name, transforms);
        }

        @Override
        TimeSeriesInstances fit(TimeSeriesInstances inst) throws Exception{
            List<TimeSeriesInstances> split = Splitter.splitTimeSeriesInstances(inst, slicingIndexes);

            List<TimeSeriesInstances> t_split = new ArrayList<TimeSeriesInstances>(layers.length);
            for (int i = 0; i < layers.length; i++) {
                System.out.println(split.get(i));
                t_split.add(layers[i].fit(split.get(i)));
            }

            return Splitter.mergeTimeSeriesInstances(t_split);
        }

        @Override
        TimeSeriesInstances predict(TimeSeriesInstances inst) throws Exception {
            List<TimeSeriesInstances> split = Splitter.splitTimeSeriesInstances(inst, slicingIndexes);

            List<TimeSeriesInstances> t_split = new ArrayList<TimeSeriesInstances>(layers.length);
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
                    output1[0][i] = TimeSeriesSummaryStatistics.mean(data.get(j).getVSliceArray(i));
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
            System.out.println(trainData);

            super.buildClassifier(trainData);
            classifier.buildClassifier(trainData);
        }

        @Override
        public double[] distributionForInstance(Instance instance) throws Exception {
            return classifier.distributionForInstance(instance);
        }
    }


}
