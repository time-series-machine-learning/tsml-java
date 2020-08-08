/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package tsml.classifiers;

import java.util.concurrent.TimeUnit;
import tsml.classifiers.distance_based.utils.classifiers.TestTimeable;

/**
 * Interface that allows the user to impose a test time contract of a classifier that
    implements this interface

    known classifiers: None
 * @author pfm15hbu
 */
public interface TestTimeContractable extends TestTimeable {
    default void setOneSecondLimit(){ setTestTimeLimit(TimeUnit.SECONDS, 1); }

    default void setOneMillisecondLimit(){ setTestTimeLimit(TimeUnit.MILLISECONDS, 1); }

    default void setSecondLimit(int t){ setTestTimeLimit(TimeUnit.SECONDS, t); }

    default void setMillisecondLimit(int t){ setTestTimeLimit(TimeUnit.MILLISECONDS, t); }

    //set any value in nanoseconds you like.
    void setTestTimeLimit(long nanos);

    //pass in an value from the TimeUnit enum and the amount of said values.
    default void setTestTimeLimit(TimeUnit unit, long amount) {
        setTestTimeLimit(amount, unit);
    }

    default void setTestTimeLimit(long amount, TimeUnit unit) {
        setTestTimeLimit(TimeUnit.NANOSECONDS.convert(amount, unit));
    }
}
