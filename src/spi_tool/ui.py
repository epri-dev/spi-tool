import panel as pn
import os

from . import _utils
from . import pages


def create_app():
    template = pn.template.MaterialTemplate(
        title="SPI-Tool Dashboard",
        # logo=os.path.join(_utils.IMAGE_FOLDER_PATH, "SPI-Tool-no-text-BW.jpg"),
        sidebar=pn.Column(sizing_mode="stretch_height"),
        main=pn.Column(pn.Column(), sizing_mode="stretch_height"),
    )
    current_page = pages.home.HomePage(template=template)
    template.sidebar.append(current_page.get_sidebar())

    modal = pn.Modal(
        pn.pane.Markdown(
            """
This software is a preproduction version which may have problems that could potentially harm your system.
To satisfy the terms and conditions of the Master License Agreement or Preproduction License Agreement between EPRI and your company, you understand what to do with this preproduction product after the preproduction review period has expired.
Reproduction or distribution of this preproduction software is in violation of the terms and conditions of the Master License Agreement, Preproduction License Agreement, or any License agreement currently in place between EPRI and your company.
Your company's funding will determine if you have the rights to the final production release of this product.
EPRI will evaluate all tester suggestions and recommendations but does not guarantee they will be incorporated into the final production product.
As a preproduction tester, you agree to provide feedback in the form requested by EPRI as a condition of obtaining the preproduction software.
        """
        ),
        name="Terms & conditions",
        max_width=800,
        max_height=400,
        open=True,
    )
    template.sidebar.append(modal)

    current_page.get_main_menu(None)

    return template
