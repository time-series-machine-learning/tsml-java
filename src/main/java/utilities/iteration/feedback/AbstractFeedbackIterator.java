package utilities.iteration.feedback;


import utilities.iteration.AbstractIterator;

public abstract class AbstractFeedbackIterator<A, B>
    extends AbstractIterator<A>
 {
    @Override
    public abstract AbstractFeedbackIterator<A, B> iterator();

    public abstract void feedback(B feedback);
}
