import panel as pn

from .. import _utils


class About(pn.viewable.Viewer):
    def __panel__(self):
        return pn.Column(
            pn.pane.Markdown(
                f"""

Stochastic Planning Inputs Tool (SPI-Tool) v{_utils.read_version_file()}

### Support

> EPRI Customer Assistance Center\\
> Phone: 800-313-3774\\
> Email: askepri@epri.com

### Ordering Information

SPI-Tool and supporting materials may be ordered from:

> EPRI, Inc.\\
> 1300 W. W.T. Harris Blvd.\\
> Charlotte, NC 28262\\
> Phone:  1-800-313-3774\\
> Email: askepri@epri.com

### Copyright

Copyright © 2025 EPRI., Inc.

EPRI reserves all rights in the Program as delivered. The Program or any portion thereof may not be reproduced in any form whatsoever except as provided by license without the written consent of EPRI. A license under EPRI's rights in the Program may be available directly from EPRI.

### Disclaimer

THIS NOTICE MAY NOT BE REMOVED FROM THE PROGRAM BY ANY USER THEREOF.

NEITHER EPRI, ANY MEMBER OF EPRI, NOR ANY PERSON OR ORGANIZATION ACTING ON BEHALF OF THEM:

1. MAKES ANY WARRANTY OR REPRESENTATION WHATSOEVER, EXPRESS OR IMPLIED, INCLUDING ANY WARRANTY OF MERCHANTABILITY OR FITNESS OF ANY PURPOSE WITH RESPECT TO THE PROGRAM; OR
2. ASSUMES ANY LIABILITY WHATSOEVER WITH RESPECT TO ANY USE OF THE PROGRAM OR ANY PORTION THEREOF OR WITH RESPECT TO ANY DAMAGES WHICH MAY RESULT FROM SUCH USE.

RESTRICTED RIGHTS LEGEND:  USE, DUPLICATION, OR DISCLOSURE BY THE UNITED STATES FEDERAL GOVERNMENT OF THE RIGHTS IN TECHNICAL DATA AND COMPUTER SOFTWARE CLAUSE IN FAR 52.227-14, ALTERNATE III IS SUBJECT TO RESTRICTION AS SET FORTH IN PARAGRAPH (3) (g) (i), WITH THE EXCEPTION OF PARAGRAPH (3) (g) (i) (4) (b).
""",
                max_width=800,
            ),
        )
