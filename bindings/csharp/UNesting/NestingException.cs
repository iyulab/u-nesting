namespace UNesting;

/// <summary>
/// Exception thrown when a nesting or packing operation fails.
/// </summary>
public class NestingException : Exception
{
    /// <summary>
    /// The native error code.
    /// </summary>
    public int ErrorCode { get; }

    /// <summary>
    /// Creates a new NestingException instance.
    /// </summary>
    /// <param name="errorCode">The native error code.</param>
    /// <param name="message">The error message.</param>
    public NestingException(int errorCode, string message)
        : base(message)
    {
        ErrorCode = errorCode;
    }

    /// <summary>
    /// Creates a new NestingException instance with an inner exception.
    /// </summary>
    /// <param name="errorCode">The native error code.</param>
    /// <param name="message">The error message.</param>
    /// <param name="innerException">The inner exception.</param>
    public NestingException(int errorCode, string message, Exception innerException)
        : base(message, innerException)
    {
        ErrorCode = errorCode;
    }
}
