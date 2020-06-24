package tsml.transformers;

import org.junit.Assert;
import tsml.classifiers.distance_based.utils.collections.intervals.Interval;
import tsml.classifiers.distance_based.utils.collections.intervals.IntervalInstance;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.List;

public class IntervalTransform extends BaseTrainableTransformer {

    private Interval interval;

    public IntervalTransform() {
        this(new Interval(0, 0));
    }

    public IntervalTransform(Interval interval) {
        setInterval(interval);
    }

    @Override public Instance transform(final Instance inst) {
        return new IntervalInstance(interval, inst);
    }

    @Override public Instances determineOutputFormat(final Instances data) throws IllegalArgumentException {
        ArrayList<Attribute> attributeList = new ArrayList<>();
        for(int i = 0; i < interval.size(); i++) {
            final int translated = interval.translate(i);
            final Attribute attribute = data.attribute(translated);
            attributeList.add(attribute);
        }
        return new Instances(data.relationName(), attributeList, 0);
    }

    public Interval getInterval() {
        return interval;
    }

    public void setInterval(final Interval interval) {
        Assert.assertNotNull(interval);
        this.interval = interval;
    }
}
