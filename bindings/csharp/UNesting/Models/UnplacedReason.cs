using System.Text.Json.Serialization;

namespace UNesting.Models;

/// <summary>Why a part is among a solve's unplaced ones.</summary>
[JsonConverter(typeof(JsonStringEnumConverter<UnplacedReason>))]
public enum UnplacedReason
{
    /// <summary>
    /// No allowed rotation of the part fits inside the sheet's bounding box less the margin:
    /// no sheet of this size can take it, so more sheets will not help.
    /// </summary>
    [JsonStringEnumMemberName("exceeds_boundary")] ExceedsBoundary,
    /// <summary>The part fits a sheet, but the layout had no room left for it.</summary>
    [JsonStringEnumMemberName("no_room")] NoRoom,
}
