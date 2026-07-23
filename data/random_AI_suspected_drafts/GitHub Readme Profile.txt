# GitHub Readme Profile

Generate dynamic GitHub stats cards in SVG format

GitHub Readme Profile is a free and open-source web service that generates dynamic SVG cards displaying a GitHub user's statistics. It is designed to be embedded in a user's GitHub profile README.md file, providing a customizable and visually consistent way to showcase GitHub activity. The project is maintained by Rangga Fajar Oktariansyah and a community of contributors.

## Features

The service generates a card that can include various metrics such as:

* Total number of repositories
* Total stars received
* Number of forks
* Total commits
* Number of pull requests (PRs) created and merged
* Number of issues opened
* Number of repositories contributed to

It also supports optional additional metrics like code reviews, closed issues, and discussion statistics.

### Customization

Users can customize the appearance and content of the card through URL parameters. Available visual customizations include:

* Themes: Predefined color schemes (e.g., dark, highcontrast, transparent).
* Colors: Individual control over title, text, icon, border, background, and stroke colors, with support for solid colors or gradient backgrounds.
* Layout: Options to flip the card layout or hide the profile image stroke.
* Data Control: Parameters to hide specific stats or show additional ones.
* Format: Output formats include SVG (default), PNG, JSON, and XML.
* Localization: Support for multiple languages via a locale parameter.

### Advanced Usage

The project includes advanced features for enhanced integration: -

* Gradient Backgrounds: Backgrounds can be set as linear gradients by specifying an angle and two colors.
* Transparency: A transparent theme or a fully transparent background color allows the card to blend seamlessly into any page background.
* GitHub Theme Context: The card can be configured to switch between light and dark themes based on the viewer's GitHub theme preference using GitHub's #gh-dark-mode-only and #gh-light-mode-only tags.
* Self-Hosting: Users can deploy their own instance of the service on Vercel, using a personal access token to fetch data and maintain control over the endpoint.

## Deployment

The project is designed to be easily deployed on Vercel. A public instance is maintained by the author at https://gh-readme-profile.vercel.app. For users wishing to host their own instance, the repository provides step-by-step instructions for forking the project, setting up a GitHub personal access token, and configuring the deployment environment.

## License and Contributions

GitHub Readme Profile is released under the MIT License, allowing for free use, modification, and distribution.

The project welcomes contributions in the form of bug fixes, new themes, translations, and feature suggestions. Contribution guidelines and a code of conduct are available in the repository.

## History and Credits

The project is a fork of tuhinpal/readme-stats-github and was inspired by anuraghazra/github-readme-stats. It builds upon these existing works to provide additional features, localization support, and customization options.