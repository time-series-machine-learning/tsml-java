package weka.classifiers.rules.sortinghandler;

/**
 *  This code is from the book:
 *
 *    Winder, R and Roberts, G (1998) <em>Developing Java
 *    Software</em>, John Wiley & Sons.
 *
 *  It is copyright (c) 1997 Russel Winder and Graham Roberts.
 */


/**
 *  A class to represent the exception of attempting to compare types
 *  of data that the <code>Comparator</code> was not designed for.
 *
 *  @see Comparator
 *  @version 1.0 2.7.1997
 *  @author Russel Winder
 */
public class ComparatorParameterErrorException
    extends RuntimeException
{
    /**
     *  The exception without a message.
     */
    public ComparatorParameterErrorException()
    {
        super() ;
    }

    /**
     *  The exception with a message.
     */
    public ComparatorParameterErrorException(final String s)
    {
        super(s) ;
    }
}

