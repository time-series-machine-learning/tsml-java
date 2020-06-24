package tsml.transformers;

import org.junit.Assert;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.List;

public class TransformPipeline extends BaseTrainableTransformer {

    private List<Transformer> transformers;

    public TransformPipeline() {
        setTransformers(new ArrayList<>());
    }

    public List<Transformer> getTransformers() {
        return transformers;
    }

    public void setTransformers(final List<Transformer> transformers) {
        Assert.assertNotNull(transformers);
        this.transformers = transformers;
    }

    @Override public void fit(final Instances data) {
        super.fit(data);
        for(Transformer transformer : transformers) {
            if(transformer instanceof TrainableTransformer) {
                ((TrainableTransformer) transformer).fit(data);
            }
        }
    }

    @Override public Instance transform(Instance inst) {
        for(Transformer transformer : transformers) {
            inst = transformer.transform(inst);
        }
        return inst;
    }

    @Override public Instances transform(Instances data) {
        for(Transformer transformer : transformers) {
            data = transformer.transform(data);
        }
        return data;
    }

    @Override public Instances determineOutputFormat(Instances data) throws IllegalArgumentException {
        for(Transformer transformer : transformers) {
            data = transformer.determineOutputFormat(data);
        }
        return data;
    }

    public boolean add(Transformer transformer) {
        return transformers.add(transformer);
    }
}
