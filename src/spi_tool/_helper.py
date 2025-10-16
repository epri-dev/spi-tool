import panel as pn


class CustomPipeline(pn.pipeline.Pipeline):
    def _update_progress(self, *args):
        super()._update_progress(*args)
        self.title.object = "## " + self._stage

    def add_stage(self, name, stage, **kwargs):
        super().add_stage(name, stage, **kwargs)

        def _update_button_to_primary(evt=None):
            if evt.new:
                self.next_button.button_type = "primary"
            else:
                self.next_button.button_type = "default"

        stage.param.watch(_update_button_to_primary, "ready", onlychanged=True)
