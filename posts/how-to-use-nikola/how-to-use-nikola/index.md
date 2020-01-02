<!--
.. title: How to use nikola?
.. slug: how-to-use-nikola
.. date: 2019-08-14 02:03:23 UTC+02:00
.. tags: python, nikola
.. category:
.. link:
.. description:
.. type: text
-->

####Content:
1. [Create new post](#new_post)
2. [Install new theme](#install_theme)
3. [Install new plugin](#install_plugin)
4. [Automatic rebuild](#automatic)

####How to generate static website using Nikola?
Nikola is static website generator. It's really easy to use.
```bash
pip install Nikola                # Install Nikola package
nikola init --demo my_first_site  # Create website directory
cd my_first_site                  # Change directory to the site
nikola build                      # Build the site
nikola serve --browser            # Run the server
```
Now website will appear in web browser.

####Create new post:<a name="new_post"></a>
Nikola support most popular Markup language. To create new post.
```bash
nikola new_post
```
Nikola will ask you for title. Assume that your title will be `HelloWorld`.
You can edit that blog at `posts/helloworld.rst`. After edit the new plog,
just run `nikola build`.

To specifiy Markdown type:
```bash
nikola new_post -f markdown
```
Now, we can edit markdown blog at `posts/helloworld.md`.

####Install a theme:<a name="install_theme"/>
We can install new theme for the website. To list avaiable themes.
```bash
nikola theme -l
```
To install `lanyon` theme.
```bash
nikola theme -i lanyon
```
Build and run server again.

####Install a plugin:<a name="install_plugin"/>
We can install new plugin for the website. To list avaiable plugins.
```bash
nikola plugin -l
```
To install `latex` plugin.
```bash
nikola plugin -i latex
```
Build and run server again.

####Automatic rebuild:<a name="automatic"/>
Build and run server again. Hmmm, I hate repticton. Nikola support nice
automatic tool watch files. It build and update server.
Still you need to install some packages first.
```bash
pip install aiohttp watchdog
nikola auto
```

