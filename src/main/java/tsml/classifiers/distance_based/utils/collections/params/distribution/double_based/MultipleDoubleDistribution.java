package tsml.classifiers.distance_based.utils.collections.params.distribution.double_based;

import java.util.List;
import java.util.Random;

public class MultipleDoubleDistribution extends DoubleDistribution {

    private List<Double> sections;

    public MultipleDoubleDistribution(List<Double> sections) {
        super(0, 0);
        setSections(sections);
    }

    @Override public Double sample() {
        final Random random = getRandom();
        final int i = random.nextInt(sections.size() - 1);
        DoubleDistribution distribution = new UniformDoubleDistribution(sections.get(i), sections.get(i + 1));
        return distribution.sample(random);
    }

    public List<Double> getSections() {
        return sections;
    }

    public void setSections(final List<Double> sections) {
        this.sections = sections;
    }
}
