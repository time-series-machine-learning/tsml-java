package tsml.graphs;

import java.util.ArrayList;
import java.util.List;

import tsml.classifiers.EnhancedAbstractClassifier;
import tsml.data_containers.TimeSeriesInstances;
import tsml.transformers.Transformer;
import weka.classifiers.AbstractClassifier;

public class Pipeline extends EnhancedAbstractClassifier{


    List<Layer> layers;

    public Pipeline(){
        layers = new ArrayList();
    }


	public void add(String name, Transformer transformer) {
        layers.add(new TransformerLayer(name, transformer));
	}

	public void add(String name, EnhancedAbstractClassifier clf){
        layers.add(new ClassifierLayer<EnhancedAbstractClassifier>(name, clf));
    }
    
    public void add(String name, AbstractClassifier clf){
        layers.add(new ClassifierLayer<AbstractClassifier>(name, clf));
    }
    
    @Override
    public void buildClassifier(TimeSeriesInstances trainData) throws Exception {
        super.buildClassifier(trainData);

        
        for ( Layer layer : layers){

        }

    }
    

    

    class Layer{
        String name;
    }

    class TransformerLayer extends Layer{
        Transformer[] transformer;

        public TransformerLayer(String name, Transformer... transformer) {
            this.name = name;
            this.transformer = transformer;
        }

        TimeSeriesInstances do(TimeSeriesInstances input){

        }
    }

    class ClassifierLayer<T extends AbstractClassifier> extends Layer{
        AbstractClassifier classifier;

        public ClassifierLayer(String name, T clf){
            this.name = name;
            this.classifier = clf;
        }
    }

    class ConcatLayer extends Layer{

    }
}