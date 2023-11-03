from rich.progress import ProgressColumn, Task, filesize
from rich.text import Text


class RateColumn(ProgressColumn):
    """Renders human readable processing rate."""

    def render(self, task: "Task") -> Text:
        """Render the speed in iterations per second."""
        speed = task.finished_speed or task.speed
        if speed is None:
            return Text("", style="progress.percentage")
        unit, suffix = filesize.pick_unit_and_suffix(
            int(speed),
            ["", "×10³", "×10⁶", "×10⁹", "×10¹²"],
            1000,
        )
        data_speed = speed / unit
        if data_speed < 1:
            data_speed = unit / speed
            return Text(f"{data_speed:.2f}{suffix} s/it", style="progress.percentage")
        else:
            return Text(f"{data_speed:.2f}{suffix} it/s", style="progress.percentage")
