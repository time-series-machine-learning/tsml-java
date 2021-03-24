/* 
 * This file is part of the UEA Time Series Machine Learning (TSML) toolbox.
 *
 * The UEA TSML toolbox is free software: you can redistribute it and/or 
 * modify it under the terms of the GNU General Public License as published 
 * by the Free Software Foundation, either version 3 of the License, or 
 * (at your option) any later version.
 *
 * The UEA TSML toolbox is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with the UEA TSML toolbox. If not, see <https://www.gnu.org/licenses/>.
 */
 
package tsml.transformers;

import org.junit.Assert;

import scala.collection.mutable.StringBuilder$;
import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;
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
    
    public void fit(TimeSeriesInstances data) {
        super.fit(data);
        
        int lastFitIndex = 0; // the index of the transformer last fitted
        for(int i = 0; i < transformers.size(); i++) {
            final Transformer transformer = transformers.get(i);
            if(transformer instanceof TrainableTransformer) {
                // in order to fit the transformer, we need to transform the data up to the point of the fittable
                // transformer
                // so transform the data from the previous transform up to this transformer
                for(int j = lastFitIndex; j < i; j++) {
                    final Transformer previous = transformers.get(j);
                    // replace the data with the data from applying a previous transform
                    data = previous.transform(data);
                }
                // this is now the most recently fit transformer in the pipeline
                lastFitIndex = i;
            }
        }
    }

    @Override public TimeSeriesInstance transform(TimeSeriesInstance inst) {
        for(Transformer transformer : transformers) {
            inst = transformer.transform(inst);
        }
        return inst;
    }

    @Override
    public Instances determineOutputFormat(Instances data) throws IllegalArgumentException {
        for (Transformer transformer : transformers) {
            data = transformer.determineOutputFormat(data);
        }
        return data;
    }

    public boolean add(Transformer transformer) {
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
    public static Transformer add(Transformer a, Transformer b) {
        if (a == null) {
            return b;
        }
        if (b == null) {
            return a;
        }
        if (a instanceof TransformPipeline) {
            ((TransformPipeline) a).add(b);
            return a;
        } else {
            return new TransformPipeline(a, b);
        }
    }

    @Override public String toString() {
        final StringBuilder sb = new StringBuilder();
        for(int i = 0; i < transformers.size() - 1; i++) {
            final Transformer transformer = transformers.get(i);
            sb.append(transformer);
            sb.append("_");
        }
        sb.append(transformers.get(transformers.size() - 1));
        return sb.toString();
    }
}
