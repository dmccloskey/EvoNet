cpack_add_install_type(recommended DISPLAY_NAME "Recommended")
cpack_add_install_type(full DISPLAY_NAME "Full")
cpack_add_install_type(minimal DISPLAY_NAME "Minimal")

cpack_add_component(share
                DISPLAY_NAME "EvoNet shared files"
                DESCRIPTION "EvoNet shared files"
                INSTALL_TYPES recommended full minimal
                )
cpack_add_component(library
                DISPLAY_NAME "Libraries"
                DESCRIPTION "Libraries"
                INSTALL_TYPES recommended full minimal
                )
cpack_add_component(applications
                DISPLAY_NAME "EvoNet binaries"
                DESCRIPTION "EvoNet binaries."
                INSTALL_TYPES recommended full minimal
                )
cpack_add_component(doc
                DISPLAY_NAME "Documentation"
                DESCRIPTION "Class and tool documentation. With tutorials."
                INSTALL_TYPES recommended full
                )