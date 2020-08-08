package tsml.transformers;

import org.junit.Assert;

import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class TransformPipeline extends BaseTrainableTransformer {

    private List<Transformer> transformers;

    public TransformPipeline() {
        this(new ArrayList<>());
    }

    public TransformPipeline(List<Transformer> transformers) {
        setTransformers(transformers);
    }

    public TransformPipeline(Transformer... transformers) {
        this(new ArrayList<>(Arrays.asList(transformers)));
    }

    public List<Transformer> getTransformers() {
        return transformers;
    }

    public void setTransformers(final List<Transformer> transformers) {
        Assert.assertNotNull(transformers);
        this.transformers = transformers;
    }

    @Override
    public void fit(Instances data) {
        super.fit(data);
        // to avoid doing needless transforms, find which of the transformers (if any)
        // need fitting and record their indices
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < transformers.size(); i++) {
            final Transformer transformer = transformers.get(i);
            if (transformer instanceof TrainableTransformer) {
                indices.add(i);
            }
        }
        // if there are transformers which need fitting
        if (!indices.isEmpty()) {
            int prevIndex = -1;
            for (int j = 0; j < indices.size(); j++) {
                // find the index of the next transformer which needs fitting
                final Integer index = indices.get(j);
                // fitTransform all transformers before that
                for (int i = prevIndex + 1; i <= index; i++) {
                    final Transformer transformer = transformers.get(i);
                    // if the transformer requires fitting then fitTransform it
                    if (transformer instanceof TrainableTransformer) {
                        // unless it's the last transformer, in which case the data will not be used
                        // afterwards so no need to transform after the fit operation
                        if (j == indices.size() - 1) {
                            ((TrainableTransformer) transformer).fit(data);
                        } else {
                            data = ((TrainableTransformer) transformer).fitTransform(data);
                        }
                    } else {
                        // the transformer cannot be fitted, so just transform the data
                        data = transformer.transform(data);
                    }
                }
                prevIndex = index;
            }
        }
    }

    @Override
    public Instance transform(Instance inst) {
        for (Transformer transformer : transformers) {
            inst = transformer.transform(inst);
        }
        return inst;
    }

    @Override
    public Instances transform(Instances data) {
        for (Transformer transformer : transformers) {
            data = transformer.transform(data);
        }
        return data;
    }

    @Override
    public Instances determineOutputFormat(Instances data) throws IllegalArgumentException {
        for (Transformer transformer : transformers) {
            data = transformer.determineOutputFormat(data);
        }
        return data;
    }

    public boolean append(Transformer transformer) {
        return transformers.add(transformer);
    }

    /**
     *
     * @param a the transformer to append to. If this is already a pipeline
     *          transformer then b is added to the list of transformers. If not, a
     *          new pipeline transformer is created and a and b are added to the
     *          list of transformers (in that order!).
     * @param b
     * @return
     */
    public static Transformer append(Transformer a, Transformer b) {
        if (a == null) {
            return b;
        }
        if (b == null) {
            return a;
        }
        if (a instanceof TransformPipeline) {
            ((TransformPipeline) a).append(b);
            return a;
        } else {
            return new TransformPipeline(a, b);
        }
    }

    @Override
    public void fit(TimeSeriesInstances data) {
        // TODO Auto-generated method stub

    }

    @Override
    public TimeSeriesInstance transform(TimeSeriesInstance inst) {
        // TODO Auto-generated method stub
        return null;
    }
}
