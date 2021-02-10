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

import experiments.data.DatasetLoading;
import org.junit.Assert;
import tsml.classifiers.distance_based.utils.collections.intervals.Interval;
import tsml.classifiers.distance_based.utils.collections.intervals.IntervalInstance;
import tsml.data_containers.TimeSeriesInstance;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;

public class IntervalTransform implements Transformer {

    private Interval interval;
    private boolean deepCopyInstances;
    private Instances header;

    public IntervalTransform() {
        this(new Interval(0, 0));
    }

    public IntervalTransform(Interval interval) {
        setInterval(interval);
        setDeepCopyInstances(false);
    }

    @Override
    public Instance transform(Instance inst) {
        if (deepCopyInstances) {
            throw new UnsupportedOperationException(); // todo
        } else if (inst instanceof IntervalInstance) {
            ((IntervalInstance) inst).setInterval(interval);
        } else {
            inst = new IntervalInstance(interval, inst);
        }
        if (inst.dataset().numAttributes() != inst.numAttributes()) {
            if (header == null || inst.dataset().numAttributes() != header.numAttributes()) {
                header = determineOutputFormat(inst.dataset());
            }
            inst.setDataset(header);
        }
        return inst;
    }

    @Override
    public Instances determineOutputFormat(Instances data) throws IllegalArgumentException {
        Assert.assertEquals(data.numAttributes() - 1, data.classIndex());
        if (deepCopyInstances) {
            throw new UnsupportedOperationException(); // todo
        } else {
            ArrayList<Attribute> attributes = new ArrayList<>(interval.size() + 1);
            for (int i = 0; i < interval.size(); i++) {
                final int j = interval.translate(i);
                Attribute attribute = data.attribute(j);
                attribute = attribute.copy(String.valueOf(j));
                attributes.add(attribute);
            }
            attributes.add((Attribute) data.classAttribute().copy());
            data = new Instances(data.relationName(), attributes, 0);
            data.setClassIndex(data.numAttributes() - 1);
        }
        return data;
    }

    public Interval getInterval() {
        return interval;
    }

    public void setInterval(final Interval interval) {
        Assert.assertNotNull(interval);
        this.interval = interval;
    }

    public boolean isDeepCopyInstances() {
        return deepCopyInstances;
    }

    public void setDeepCopyInstances(final boolean deepCopyInstances) {
        this.deepCopyInstances = deepCopyInstances;
    }

    public static void main(String[] args) throws Exception {
        final Instances instances = DatasetLoading.loadGunPoint();
        final Instance instance = instances.get(0);
        final IntervalInstance intervalInstance = new IntervalInstance(new Interval(10, 5), instance);
        System.out.println(intervalInstance.toString());
    }

    @Override
    public TimeSeriesInstance transform(TimeSeriesInstance inst) {
        // TODO Auto-generated method stub
        return null;
    }
}
