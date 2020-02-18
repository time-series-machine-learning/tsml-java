package utilities;

import java.lang.annotation.Retention;
import java.lang.annotation.Target;

import static java.lang.annotation.ElementType.*;
import static java.lang.annotation.RetentionPolicy.CLASS;

@Retention(CLASS)
@Target({FIELD, METHOD, PARAMETER, LOCAL_VARIABLE})
public @interface NotNull {String value() default "";}
