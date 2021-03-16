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

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import experiments.data.DatasetLoading;
import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;

import org.junit.Assert;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;
import tsml.data_containers.utilities.Converter;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Purpose: cache the filtering operation using a map. Note, the instances must
 * be hashed first to use the cache reliably otherwise issues occur with
 * instance copying changing the hashcode due to memory locations.
 * <p>
 * Contributors: goastler, abostrom
 */
public class CachedTransformer extends BaseTrainableTransformer {

    // the filter to cache the output of
    private Transformer transformer;
    // whether to only cache instances from the fit() call OR all instances handed
    // to the transform method
    private boolean cacheFittedDataOnly;

    // the cache to store instances against their corresponding transform output
    private Map<TimeSeriesInstance, TimeSeriesInstance> tsCache; // use object as key so we can accept either inst or tsinst
    private Map<Instance, Instance> arffCache;

    public CachedTransformer(final Transformer transformer) {
        setTransformer(transformer);
        setCacheFittedDataOnly(true);
        reset();
    }

    public boolean isCacheFittedDataOnly() {
        return cacheFittedDataOnly;
    }

    public void setCacheFittedDataOnly(final boolean cacheFittedDataOnly) {
        this.cacheFittedDataOnly = cacheFittedDataOnly;
    }

    public void reset() {
        super.reset();
        tsCache = new HashMap<>();
        arffCache = new HashMap<>();
    }

    @Override
    public void fit(final Instances data) {
        super.fit(data);
        if(transformer instanceof TrainableTransformer) {
            ((TrainableTransformer) transformer).fit(data);
        }
        for (final Instance instance : data) {
            arffCache.put(instance, null);
        }
    }

    @Override
    public void fit(final TimeSeriesInstances data) {
        super.fit(data);
        if(transformer instanceof TrainableTransformer) {
            ((TrainableTransformer) transformer).fit(data);
        }
        for (final TimeSeriesInstance instance : data) {
            tsCache.put(instance, null);
        }
    }

    @Override
    public String toString() {
        return transformer.getClass().getSimpleName();
    }

    public void setTransformer(final Transformer transformer) {
        Assert.assertNotNull(transformer);
        this.transformer = transformer;
    }

    @Override
    public TimeSeriesInstance transform(TimeSeriesInstance inst) {
        if(!isFit()) {
            throw new IllegalStateException("must be fitted first");
        }
        TimeSeriesInstance transformed = tsCache.get(inst);
        if(transformed == null) {
            transformed = transformer.transform(inst);
            if(!cacheFittedDataOnly || tsCache.containsKey(inst)) {
                tsCache.put(inst, transformed);
            }
        }
        return transformed;
    }

    @Override public Instance transform(final Instance inst) {
        if(!isFit()) {
            throw new IllegalStateException("must be fitted first");
        }
        Instance transformed = arffCache.get(inst);
        if(transformed == null) {
            transformed = transformer.transform(inst);
            if(!cacheFittedDataOnly || arffCache.containsKey(inst)) {
                arffCache.put(inst, transformed);
            }
        }
        return transformed;
    }

    @Override
    public Instances determineOutputFormat(final Instances data) throws IllegalArgumentException {
        return transformer.determineOutputFormat(data);
    }

    public Transformer getTransformer() {
        return transformer;
    }

    @Override
    public void setParams(final ParamSet paramSet) throws Exception {
        super.setParams(paramSet);
        setTransformer(paramSet.get(TRANSFORMER_FLAG, getTransformer()));
    }

    @Override
    public ParamSet getParams() {
        return new ParamSet().add(TRANSFORMER_FLAG, transformer);
    }

    public static final String TRANSFORMER_FLAG = "f";

    public static void main(String[] args) throws Exception {
        final CachedTransformer ct = new CachedTransformer(new Derivative());
        final List<TimeSeriesInstances> data =
                Arrays.stream(DatasetLoading.sampleGunPoint(0)).map(Converter::fromArff).collect(Collectors.toList());
        ct.fit(data.get(0));
        ct.transform(data.get(1).get(0));
        ct.transform(data.get(0).get(0));
    }
}
